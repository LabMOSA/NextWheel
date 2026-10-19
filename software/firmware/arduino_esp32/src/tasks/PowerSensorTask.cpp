#include "PowerSensorTask.h"
#include "NextWheelApp.h"

PowerSensorTask::PowerSensorTask() : SensorTask("PowerSensorTask", POWER_SENSOR_TASK_STACK_SIZE) {}

void PowerSensorTask::run(void* app)
{
    Serial.printf("PowerSensorTask::run Priority: %li Core: %li \n", uxTaskPriorityGet(NULL), xPortGetCoreID());

    m_power.begin();
    TickType_t lastGeneration = xTaskGetTickCount();
    PowerDataFrame frame;
    uint32_t low_battery_counter = 0;

    while (1)
    {
        // First empty the command queue (timeout=0, not waiting)
        // Loop while we have BASE_TASK_COMMAND_NONE --> 0
        while (Task::BaseTaskCommand command = dequeueBaseCommand(0))
        {
            switch (command)
            {
                case Task::BASE_TASK_COMMAND_NONE:
                    Serial.println("PowerSensorTask::run: BASE_TASK_COMMAND_NONE");
                    break;
                case Task::BASE_TASK_CONFIG_UPDATED:
                    Serial.println("PowerSensorTask::run: BASE_TASK_CONFIG_UPDATED");
                    break;
                default:
                    Serial.print("PowerSensorTask::run: Unknown command: ");
                    Serial.println(command);
                    break;
            }
        }

        // 1000 ms task
        vTaskDelayUntil(&lastGeneration, 1000 / portTICK_RATE_MS);

        // Update values
        m_power.update(frame);

        // TODO Verify low battery threshold, will beep every 60 seconds
        // Low battery free running counter, increments every second
        if (m_power.isLowPower() && (low_battery_counter++ % 60 == 0))
        {
            Serial.printf("PowerSensorTask::run: Low battery. Voltage: %f\n", frame.getVoltage());
            // Play low battery sound
            NextWheelApp::instance()->playLowBatterySound();
        }

        // Send data to registered queues
        sendData(frame);
    }
}
