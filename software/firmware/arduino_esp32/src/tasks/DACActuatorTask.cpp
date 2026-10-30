#include "DACActuatorTask.h"


// [f Hz, duration ms, f Hz, duration ms, ...]
const uint32_t DACActuatorTask::START_RECORDING_SOUND[SOUND_QUEUE_MESSAGE_SIZE] =
    {750, 150, 1000, 150, 1500, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
const uint32_t DACActuatorTask::STOP_RECORDING_SOUND[SOUND_QUEUE_MESSAGE_SIZE] =
    {1500, 150, 1000, 150, 750, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const uint32_t DACActuatorTask::START_STREAMING_SOUND[SOUND_QUEUE_MESSAGE_SIZE] =
    {1000, 150, 1500, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
const uint32_t DACActuatorTask::STOP_STREAMING_SOUND[SOUND_QUEUE_MESSAGE_SIZE] =
    {1500, 150, 1000, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const uint32_t DACActuatorTask::LOW_BATTERY_SOUND[16] =
    {1000, 100, 0, 100, 1000, 100, 0, 100, 1000, 100, 0, 100, 1000, 100, 0, 100};


/**
 * @brief Scale data to 16bit/32bit for I2S DMA output.
 *        DAC can only output 8bit data value.
 *        I2S DMA will still send 16 bit or 32bit data, the highest 8bit contains DAC data.
 */
int example_i2s_dac_data_scale(uint8_t* d_buff, uint8_t* s_buff, uint32_t len)
{
    uint32_t j = 0;
    for (int i = 0; i < len; i++)
    {
        d_buff[j++] = 0;
        d_buff[j++] = s_buff[i];
    }
    return (len * 2);
}


DACActuatorTask::DACActuatorTask(const char* name, uint32_t stackSize, uint8_t priority)
    : Task(name, stackSize, priority)
{
    m_soundQueue = xQueueCreate(10, SOUND_QUEUE_MESSAGE_SIZE * sizeof(uint32_t));

#ifdef _USE_INTERNAL_DAC_
    m_dac.setup();
#else
    // Standard PWM init
    // enable PWM ouput on GPIO25
    ledcSetup(0, 2000, 8);
    ledcAttachPin(PIN_SPI_CS2, 0);
    ledcWrite(0, 0);  // Ch:0, Duty:0

#endif
}

bool DACActuatorTask::dequeueSound(uint32_t* sound, unsigned long timeout)
{
    if (xQueueReceive(m_soundQueue, sound, timeout) != pdTRUE)
    {
        return false;
    }
    return true;
}

void DACActuatorTask::playStartRecordingSound(bool from_isr)
{
    if (from_isr)
        xQueueSendFromISR(m_soundQueue, START_RECORDING_SOUND, 0);
    else
        xQueueSend(m_soundQueue, START_RECORDING_SOUND, 0);
}

void DACActuatorTask::playStopRecordingSound(bool from_isr)
{
    if (from_isr)
        xQueueSendFromISR(m_soundQueue, STOP_RECORDING_SOUND, 0);
    else
        xQueueSend(m_soundQueue, STOP_RECORDING_SOUND, 0);
}

void DACActuatorTask::playStartStreamingSound(bool from_isr)
{
    if (from_isr)
        xQueueSendFromISR(m_soundQueue, START_STREAMING_SOUND, 0);
    else
        xQueueSend(m_soundQueue, START_STREAMING_SOUND, 0);
}

void DACActuatorTask::playStopStreamingSound(bool from_isr)
{
    if (from_isr)
        xQueueSendFromISR(m_soundQueue, STOP_STREAMING_SOUND, 0);
    else
        xQueueSend(m_soundQueue, STOP_STREAMING_SOUND, 0);
}

void DACActuatorTask::playLowBatterySound(bool from_isr)
{
    if (from_isr)
        xQueueSendFromISR(m_soundQueue, LOW_BATTERY_SOUND, 0);
    else
        xQueueSend(m_soundQueue, LOW_BATTERY_SOUND, 0);
}

void DACActuatorTask::run(void*)
{
    Serial.printf("DACActuatorTask::run Priority: %li Core: %li \n", uxTaskPriorityGet(NULL), xPortGetCoreID());

#ifdef _USE_INTERNAL_DAC_
    const uint32_t sampling_rate = m_dac.getSampleRate();
    const uint32_t frequency = 400;
    size_t index = 0;
    uint8_t stereo_buffer[4] = {0, 0, 0, 0};  // Right, left

#else
    // Standard PWM
    TickType_t lastGeneration = xTaskGetTickCount();
#endif

    // Common part, read base commnd

    while (1)
    {
        // First empty the command queue (timeout=0, not waiting)
        // Loop while we have BASE_TASK_COMMAND_NONE --> 0
        while (Task::BaseTaskCommand command = dequeueBaseCommand(0))
        {
            switch (command)
            {
                case Task::BASE_TASK_COMMAND_NONE:
                    Serial.println("DACActuatorTask::run: BASE_TASK_COMMAND_NONE");
                    break;
                case Task::BASE_TASK_CONFIG_UPDATED:
                    Serial.println("DACActuatorTask::run: BASE_TASK_CONFIG_UPDATED");
                    break;
                default:
                    Serial.print("DACActuatorTask::run: Unknown command: ");
                    Serial.println(command);
                    break;
            }
        }

#ifdef _USE_INTERNAL_DAC_

        uint16_t sample =
            uint16_t(120.0 * sin(2 * M_PI * (double)frequency * (double)index++ / (double)sampling_rate) + 120.0);
        // uint16_t sample = 0;
        // Serial.printf("DACActuatorTask::run sample: i: %i %u \n", index, sample);

        // DAC will only use the MSB
        //  Right channel
        stereo_buffer[0] = 0;
        stereo_buffer[1] = (sample) & 0xFF;

        // Left channel
        // stereo_buffer[2] = 0;
        // stereo_buffer[3] = sample & 0xFF;

        // Write to DAC (will wait if buffer is full)
        //  Only using right channel
        m_dac.writeFrame(stereo_buffer, sizeof(stereo_buffer) / 2);

#else
        // Sound will be stored in this array
        uint32_t sound_to_play[SOUND_QUEUE_MESSAGE_SIZE] = {0};

        // Dequeue sound message from queue, with timeout of 1000 ms
        if (dequeueSound(&sound_to_play[0], 1000 / portTICK_RATE_MS))
        {
            for (int i = 0; i < SOUND_QUEUE_MESSAGE_SIZE; i += 2)
            {
                uint32_t frequency = sound_to_play[i];
                uint32_t duration = sound_to_play[i + 1];

                // Serial.printf("Playing sound: %i %i \n", frequency, duration);
                if (frequency == 0)
                {
                    // No sound
                    ledcWrite(0, 0);  // Ch:0, Duty: 0/256 (0%)
                }
                else
                {
                    // Valid sound
                    ledcSetup(0, frequency, 8);  // channel 0, 2000 Hz, 8-bit resolution
                    ledcWrite(0, 128);  // Ch:0, Duty: 128/256 (50%)
                }

                vTaskDelay(duration / portTICK_PERIOD_MS);
            }
            ledcWrite(0, 0);  // Ch:0, Duty: 0/256 (0%)
        }
        else
        {
            // Serial.println("Nothing to play");
            ledcWrite(0, 0);  // Ch:0, Duty: 0/256 (0%)
        }
#endif

    }  // while(1)
}  // run
