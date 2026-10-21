#include "QuadEncoderSensorTask.h"
#include "config/GlobalConfig.h"
#include "data/QuadEncoderDataFrame.h"

namespace NextWheelInterrupts
{
    SemaphoreHandle_t g_quad_encoder_semaphore;
    void IRAM_ATTR quad_encoder_sensor_task_timer_interrupt()
    {
        xSemaphoreGiveFromISR(NextWheelInterrupts::g_quad_encoder_semaphore, NULL);
    }
}  // namespace NextWheel


QuadEncoderSensorTask::QuadEncoderSensorTask() : SensorTask("QuadEncoderSensorTask")
{
    NextWheelInterrupts::g_quad_encoder_semaphore = xSemaphoreCreateCounting(1, 0);
}


void QuadEncoderSensorTask::run(void* app)
{
    Serial.printf("QuadEncoderSensorTask::run Priority: %li Core: %li \n", uxTaskPriorityGet(NULL), xPortGetCoreID());

    TickType_t lastGeneration = xTaskGetTickCount();

    // Setup encoder
    m_encoder.attachFullQuad(PIN_QUAD_ENCODER_A, PIN_QUAD_ENCODER_B);
    m_encoder.clearCount();

    auto encoder_timer = timerBegin(1, 80, true);  // count up. 80 prescaler = 1us resolution
    timerAttachInterrupt(
        encoder_timer,
        &NextWheelInterrupts::quad_encoder_sensor_task_timer_interrupt,
        false);  // Attach interrupt function
    timerAlarmWrite(
        encoder_timer,
        1000000 / GlobalConfig::instance().get_encoder_sample_rate(),
        true);  // us timer calculation
    timerAlarmEnable(encoder_timer);


    QuadEncoderDataFrame frame;

    while (1)
    {
        // First empty the command queue (timeout=0, not waiting)
        // Loop while we have BASE_TASK_COMMAND_NONE --> 0
        while (Task::BaseTaskCommand command = dequeueBaseCommand(0))
        {
            switch (command)
            {
                case Task::BASE_TASK_COMMAND_NONE:
                    Serial.println("QuadEncoderSensorTask::run: BASE_TASK_COMMAND_NONE");
                    break;
                case Task::BASE_TASK_CONFIG_UPDATED:
                    Serial.println("QuadEncoderSensorTask::run: BASE_TASK_CONFIG_UPDATED");
                    timerAlarmDisable(encoder_timer);
                    // update sampling rate
                    timerAlarmWrite(
                        encoder_timer,
                        1000000 / GlobalConfig::instance().get_encoder_sample_rate(),
                        true);  // us timer calculation

                    timerAlarmEnable(encoder_timer);
                    break;
                default:
                    Serial.print("QuadEncoderSensorTask::run: Unknown command: ");
                    Serial.println(command);
                    break;
            }
        }

        // Encoder update will be triggered by timer interrupt
        xSemaphoreTake(NextWheelInterrupts::g_quad_encoder_semaphore, portMAX_DELAY);

        // Save encoder value
        int64_t count = m_encoder.getCount();

        // Reset counter
        // m_encoder.clearCount();

        // Serial.print("encoder counts: ");
        // Serial.println(count);

        // Update timestamp and value
        frame.setTimestamp(DataFrame::getCurrentTimeStamp());
        frame.setCount(count);
        // Send data to registered queues
        sendData(frame);
    }

    timerAlarmDisable(encoder_timer);
    timerEnd(encoder_timer);
}
