#include <Arduino.h>
#include <NextWheel.h>
#include "NextWheelApp.h"
#include <esp_adc_cal.h>
#include <driver/adc.h>
#include <WiFi.h>


extern "C"
{
#include <esp_wifi.h>
}

void setup()
{
    esp_wifi_set_ps(WIFI_PS_NONE);

    // Serial must be initialized for prints to work
    Serial.begin(115200);

    // Global app object
    // Needs to be initialized after Arduino objects are created and ready.
    // We allocate the app in the setup function to make sure it is initialized.

    // Setup app
    NextWheelApp::instance()->begin();

    // Start tasks
    NextWheelApp::instance()->start();
}

void signal_start_end_of_ip_signal(TickType_t& lastGeneration)
{
    // Signal Start of IP address
    NextWheelApp::instance()->getLEDS().setLED1(false);
    NextWheelApp::instance()->getLEDS().setLED2(false);
    vTaskDelayUntil(&lastGeneration, 1000 / portTICK_RATE_MS);
    NextWheelApp::instance()->getLEDS().setLED1(true);
    NextWheelApp::instance()->getLEDS().setLED2(true);
    vTaskDelayUntil(&lastGeneration, 1000 / portTICK_RATE_MS);
    NextWheelApp::instance()->getLEDS().setLED1(false);
    NextWheelApp::instance()->getLEDS().setLED2(false);
}

void signal_start_of_digit_with_led_1(TickType_t& lastGeneration)
{
    NextWheelApp::instance()->getLEDS().setLED1(true);
    vTaskDelayUntil(&lastGeneration, 100 / portTICK_RATE_MS);
    NextWheelApp::instance()->getLEDS().setLED1(false);
    vTaskDelayUntil(&lastGeneration, 200 / portTICK_RATE_MS);
}

void signal_digit_value_with_led_2(TickType_t& lastGeneration, uint8_t val)
{
    // Serial.print("Signaling: "); Serial.println(val);
    if (val == 0)
    {
        vTaskDelayUntil(&lastGeneration, 300 / portTICK_RATE_MS);
    }
    else
    {
        for (auto j = 0; j < val; j++)
        {
            // Signal changing Digit
            NextWheelApp::instance()->getLEDS().setLED2(true);
            vTaskDelayUntil(&lastGeneration, 100 / portTICK_RATE_MS);
            NextWheelApp::instance()->getLEDS().setLED2(false);
            vTaskDelayUntil(&lastGeneration, 200 / portTICK_RATE_MS);
        }
    }
}

void loop()
{
    // Note Arduino (loop) runs on core 1
    // WiFi, BLE runs on core 0
    // Default loop priority is TASK_PRIORITY_LOWEST (1)

    // Set Task To High Priority!
    // Buttons are very high priority to start / stop recordings...
    vTaskPrioritySet(NULL, TASK_PRIORITY_HIGHEST);
    Serial.print("Main Loop: priority = ");
    Serial.println(uxTaskPriorityGet(NULL));

    Serial.print("Main Loop: Executing on core ");
    Serial.println(xPortGetCoreID());

    String currentIP = WiFi.localIP().toString();

    TickType_t lastGeneration = xTaskGetTickCount();

    TaskHandle_t idleTaskCore0, idleTaskCore1;
    idleTaskCore0 = xTaskGetIdleTaskHandleForCPU(0);
    idleTaskCore1 = xTaskGetIdleTaskHandleForCPU(1);

    while (1)
    {
        struct timeval timeval_now;
        gettimeofday(&timeval_now, NULL);

        // Verify that the IP has changed...
        // Signal IP with LEDS... Not optimal but for works for now.
        if (currentIP != WiFi.localIP().toString())
        {
            // Store new IP
            currentIP = WiFi.localIP().toString();
            Serial.print("IP address changed: ");
            Serial.println(currentIP);


            signal_start_end_of_ip_signal(lastGeneration);

            // Iterate through each byte of the IP address
            for (auto byte_nb = 0; byte_nb < 4; byte_nb++)
            {
                uint8_t ip_val = WiFi.localIP()[byte_nb];

                // Signal 100th
                signal_start_of_digit_with_led_1(lastGeneration);
                signal_digit_value_with_led_2(lastGeneration, ip_val / 100);

                // Signal 10th
                signal_start_of_digit_with_led_1(lastGeneration);
                signal_digit_value_with_led_2(lastGeneration, (ip_val % 100) / 10);

                // Signal unit
                signal_start_of_digit_with_led_1(lastGeneration);
                signal_digit_value_with_led_2(lastGeneration, ip_val % 10);


                // Pause for each digit (.)
                vTaskDelayUntil(&lastGeneration, 1000 / portTICK_RATE_MS);
            }


            // End of IP address
            signal_start_end_of_ip_signal(lastGeneration);
        }
        else
        {
            // LED1 - Recording
            if (NextWheelApp::instance()->isRecording())
            {
                NextWheelApp::instance()->getLEDS().toggleLED1();
            }
            else
            {
                NextWheelApp::instance()->getLEDS().setLED1(false);
            }

            // LED 2 - Streaming (UDP)
            if (NextWheelApp::instance()->isStreaming())
            {
                NextWheelApp::instance()->getLEDS().toggleLED2();
            }
            else
            {
                NextWheelApp::instance()->getLEDS().setLED2(true);
            }

            if (NextWheelApp::instance()->button1Pressed())
            {
                Serial.println("Button 1 pressed");
                NextWheelApp::instance()->startRecording();
            }

            if (NextWheelApp::instance()->button2Pressed())
            {
                Serial.println("Button 2 pressed");
                NextWheelApp::instance()->stopRecording();
            }
        }

        auto freeHeap = ESP.getFreeHeap();
        if (freeHeap < 50000)
        {
            Serial.print("WARNING : Free heap size: ");
            Serial.println(ESP.getFreeHeap());
        }


        // CPU Usage (approximation)

        // IDLE loop.
        // 250 ms task
        vTaskDelayUntil(&lastGeneration, 250 / portTICK_RATE_MS);
    }
}
