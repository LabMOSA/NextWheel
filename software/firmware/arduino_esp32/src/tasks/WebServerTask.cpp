#include "WebServerTask.h"
#include "config/GlobalConfig.h"
#include "NextWheelApp.h"
#include <esp_task_wdt.h>
#include <WiFi.h>
#include <config/GlobalConfig.h>
#include <state/SystemState.h>
#include <cJSON.h>
#include <SD_MMC.h>


WebServerTask::WebServerTask()
    : WorkerTask("WebServerTask", WEB_SERVER_STACK_SIZE),
      m_webServer(80)
{
}


void WebServerTask::run(void* app)
{
    Serial.printf("WebServerTask::run Priority: %li Core: %li \n", uxTaskPriorityGet(NULL), xPortGetCoreID());

    WiFi.begin(WIFI_DEFAULT_SSID, WIFI_DEFAULT_PASSWORD);
    // put your setup code here, to run once:
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
    }
    Serial.print("Connected to WiFi IP: ");
    Serial.println(WiFi.localIP());


    TickType_t lastGeneration = xTaskGetTickCount();

    setupRESTAPI();
    m_webServer.begin();

    while (1)
    {
        // Serial.println("WebServerTask::run: m_webServer.loop()");
        m_webServer.handleClient();

        // First empty the command queue (timeout=0, not waiting)
        // Loop while we have BASE_TASK_COMMAND_NONE --> 0
        while (Task::BaseTaskCommand command = dequeueBaseCommand(0))
        {
            switch (command)
            {
                case Task::BASE_TASK_COMMAND_NONE:
                    Serial.println("WebServerTask::run: BASE_TASK_COMMAND_NONE");
                    break;
                case Task::BASE_TASK_CONFIG_UPDATED:
                {
                    Serial.println("WebServerTask::run: BASE_TASK_CONFIG_UPDATED");

                    // We should update the configuration info from now
                    ConfigDataFrame configDataFrame(GlobalConfig::instance().get());

                    // Send to UDP destination (does nothing if not streaming)
                    m_udpStreamer.sendFrame(configDataFrame);

                    // Empty the queue (with old config!)
                    while (DataFramePtr dataPtr = dequeue(0))
                    {
                        delete dataPtr;
                    }
                }
                break;

                default:
                    Serial.print("WebServerTask::run: Unknown command: ");
                    Serial.println(command);
                    break;
            }
        }

        // Then dequeue all data frames and send them to the websocket
        size_t total_payload_size = 0;
        std::list<DataFramePtr> dataPtrs;

        // Dequeue all values (one shot,), no timeout
        // Make sure we have maximum payload of approximately 1400 bytes
        // Otherwize the IP stack will have to fragment the packet and we will have a lot of overhead
        while (DataFramePtr dataPtr = dequeue(50 / portTICK_RATE_MS))
        {
            total_payload_size += dataPtr->getTotalSize();
            dataPtrs.push_back(dataPtr);

            // Enough data for one transmission ?
            // Size 1400 is selected to avoid IP fragmentation
            if (total_payload_size > 1400)
            {
                break;
            }
        }

        if (dataPtrs.size() > 0)
        {
            // uint8_t* super_frame = new uint8_t[total_payload_size + DataFrame::HEADER_SIZE];
            // Serial.printf("Should allocate %i bytes\n", total_payload_size + DataFrame::HEADER_SIZE);
            uint8_t super_frame[total_payload_size + DataFrame::HEADER_SIZE];  // Allocate on stack
            uint64_t timestamp = DataFrame::getCurrentTimeStamp();

            super_frame[0] = 255;
            memcpy(super_frame + 1, &timestamp, sizeof(uint64_t));
            super_frame[9] = dataPtrs.size();

            size_t offset = DataFrame::HEADER_SIZE;
            for (auto dataPtr : dataPtrs)
            {
                dataPtr->serialize(super_frame + offset, dataPtr->getTotalSize());
                offset += dataPtr->getTotalSize();
                delete dataPtr;
            }

            // Send to UDP destination (does nothing if not streaming)
            m_udpStreamer.send(super_frame, total_payload_size + DataFrame::HEADER_SIZE);
        }
    }  // while(1)
}

bool WebServerTask::isUDPStreaming()
{
    return m_udpStreamer.isStreaming();
}

bool WebServerTask::startUDPStreaming(uint32_t address, uint16_t port)
{
    if (m_udpStreamer.isStreaming())
    {
        return false;
    }

    // Destination is stored in the config, UDPStreamer reads it when sending
    GlobalConfig::instance().set_udp_broadcast_address(address);
    GlobalConfig::instance().set_udp_broadcast_port(port);

    if (!m_udpStreamer.start())
    {
        return false;
    }

    Serial.printf("UDP streaming to %s:%u\n",
                  IPAddress(GlobalConfig::instance().get_udp_broadcast_address()).toString().c_str(),
                  GlobalConfig::instance().get_udp_broadcast_port());

    SystemState::instance().getState().streaming = true;
    NextWheelApp::instance()->playStartStreamingSound();

    // Send the current configuration first, clients need it to decode data frames
    ConfigDataFrame configDataFrame(GlobalConfig::instance().get());
    m_udpStreamer.sendFrame(configDataFrame);

    NextWheelApp::instance()->registerSensorTasksToWebServer();
    return true;
}

bool WebServerTask::stopUDPStreaming()
{
    if (!m_udpStreamer.isStreaming())
    {
        return false;
    }

    NextWheelApp::instance()->unregisterSensorTasksFromWebServer();
    Serial.printf("UDP streaming stopped, dropped packets: %u\n", m_udpStreamer.getDroppedPackets());
    m_udpStreamer.stop();

    SystemState::instance().getState().streaming = false;
    NextWheelApp::instance()->playStopStreamingSound();
    return true;
}

void WebServerTask::setupRESTAPI()
{
    m_webServer.onNotFound(
        [this]()
        {
            Serial.println("WebServerTask::setupRESTAPI: onNotFound");
            String message = "Not Found\n\n";
            message += "URI: ";
            message += m_webServer.uri();
            message += "\nMethod: ";
            message += (m_webServer.method() == HTTP_GET) ? "GET" : "POST";
            message += "\nArguments: ";
            message += m_webServer.args();
            message += "\n";
            for (uint8_t i = 0; i < m_webServer.args(); i++)
            {
                message += " " + m_webServer.argName(i) + ": " + m_webServer.arg(i) + "\n";
            }
            m_webServer.send(404, "text/plain", message);
        });


    m_webServer.on(
        "/",
        HTTP_GET,
        [this]()
        {
            Serial.println("WebServerTask::setupRESTAPI: /");
            String message = "Hello from NextWheel REST API Server !";
            m_webServer.send(200, "text/plain", message);
        });

    m_webServer.on(
        "/config_set_time",
        HTTP_POST,
        [this]()
        {
            Serial.println("post: /config_set_time");
            if (m_webServer.hasArg("time"))
            {
                String time = m_webServer.arg("time");
                // Set the time
                NextWheelApp::instance()->setTime(time);

                // Printing current time
                struct timeval current_time;
                gettimeofday(&current_time, NULL);
                struct tm* time_info = localtime(&current_time.tv_sec);
                Serial.print("Current time: ");
                Serial.println(String(asctime(time_info)));

                m_webServer.send(200, "text/plain", "OK");
            }
            else
            {
                m_webServer.send(400, "text/plain", "Missing time argument");
            }
        });

    m_webServer.on(
        "/config_update",
        HTTP_POST,
        [this]()
        {
            Serial.println("post: /config_update");

            if (m_webServer.hasArg("accelerometer_precision") && m_webServer.hasArg("gyrometer_precision") &&
                m_webServer.hasArg("imu_sampling_rate") && m_webServer.hasArg("adc_sampling_rate") &&
                m_webServer.hasArg("encoder_sampling_rate"))
            {
                GlobalConfig::instance().set_accel_range(m_webServer.arg("accelerometer_precision").toInt());
                GlobalConfig::instance().set_gyro_range(m_webServer.arg("gyrometer_precision").toInt());
                GlobalConfig::instance().set_imu_sample_rate(m_webServer.arg("imu_sampling_rate").toInt());
                GlobalConfig::instance().set_adc_sample_rate(m_webServer.arg("adc_sampling_rate").toInt());
                GlobalConfig::instance().set_encoder_sample_rate(m_webServer.arg("encoder_sampling_rate").toInt());

                // Print current config
                Serial.println("Current config:");
                Serial.printf("Accelerometer precision: %d\n", GlobalConfig::instance().get_accel_range());
                Serial.printf("Gyrometer precision: %d\n", GlobalConfig::instance().get_gyro_range());
                Serial.printf("IMU sampling rate: %d\n", GlobalConfig::instance().get_imu_sample_rate());
                Serial.printf("ADC sampling rate: %d\n", GlobalConfig::instance().get_adc_sample_rate());
                Serial.printf("Encoder sampling rate: %d\n", GlobalConfig::instance().get_encoder_sample_rate());

                m_webServer.send(200, "text/plain", "OK");
            }
            else
            {
                m_webServer.send(
                    400,
                    "text/plain",
                    "Missing arguments (accelerometer_precision, gyrometer_precision, imu_sampling_rate, "
                    "adc_sampling_rate, encoder_sampling_rate)");
            }
        });


    m_webServer.on(
        "/config",
        HTTP_GET,
        [this]()
        {
            Serial.println("get: /config");

            cJSON* root = cJSON_CreateObject();
            cJSON_AddNumberToObject(root, "accelerometer_precision", GlobalConfig::instance().get_accel_range());
            cJSON_AddNumberToObject(root, "gyrometer_precision", GlobalConfig::instance().get_gyro_range());
            cJSON_AddNumberToObject(root, "imu_sampling_rate", GlobalConfig::instance().get_imu_sample_rate());
            cJSON_AddNumberToObject(root, "adc_sampling_rate", GlobalConfig::instance().get_adc_sample_rate());
            cJSON_AddNumberToObject(root, "encoder_sampling_rate", GlobalConfig::instance().get_encoder_sample_rate());
            /*
            cJSON_AddStringToObject(
                root, "udp_broadcast_address",
                IPAddress(GlobalConfig::instance().get_udp_broadcast_address()).toString().c_str());
            cJSON_AddNumberToObject(root, "udp_broadcast_port", GlobalConfig::instance().get_udp_broadcast_port());
            */
            String json(cJSON_Print(root));

            // Free memory
            cJSON_free(root);

            Serial.println(json);
            m_webServer.send(200, "application/json", json);
        });

    m_webServer.on(
        "/system_state",
        HTTP_GET,
        [this]()
        {
            Serial.println("get: /system_state");

            cJSON* root = cJSON_CreateObject();
            cJSON_AddNumberToObject(root, "recording", SystemState::instance().getState().recording);
            cJSON_AddNumberToObject(root, "streaming", SystemState::instance().getState().streaming);
            cJSON_AddStringToObject(root, "filename", SystemState::instance().getState().filename.c_str());
            String json(cJSON_Print(root));

            // Free memory
            cJSON_free(root);

            Serial.println(json);

            m_webServer.send(200, "application/json", json);
        });

    m_webServer.on(
        "/start_recording",
        HTTP_GET,
        [this]()
        {
            Serial.println("get: /start_recording");
            if (SystemState::instance().getState().recording)
            {
                // Already recording
                m_webServer.send(400, "text/plain", "Already recording");
            }
            else
            {
                // Start Recording
                NextWheelApp::instance()->startRecording();
                m_webServer.send(200, "text/plain", "OK");
            }
        });

    m_webServer.on(
        "/stop_recording",
        HTTP_GET,
        [this]()
        {
            if (SystemState::instance().getState().recording)
            {
                // Start Recording
                NextWheelApp::instance()->stopRecording();
                m_webServer.send(200, "text/plain", "OK");
            }
            else
            {
                // Not recording
                m_webServer.send(400, "text/plain", "Not recording");
            }
        });


    m_webServer.on(
        "/start_streaming",
        HTTP_GET,
        [this]()
        {
            Serial.println("get: /start_streaming");
            if (!m_webServer.hasArg("address") || !m_webServer.hasArg("port"))
            {
                m_webServer.send(400, "text/plain", "Missing arguments (address, port)");
                return;
            }
            IPAddress udp_address;
            if (!udp_address.fromString(m_webServer.arg("address")))
            {
                m_webServer.send(400, "text/plain", "Invalid address (expected x.x.x.x)");
                return;
            }
            long udp_port = m_webServer.arg("port").toInt();
            if (udp_port <= 0 || udp_port >= 0xFFFF)
            {
                m_webServer.send(400, "text/plain", "Invalid port");
                return;
            }

            if (startUDPStreaming(uint32_t(udp_address), udp_port))
            {
                m_webServer.send(200, "text/plain", "OK");
            }
            else
            {
                m_webServer.send(400, "text/plain", "Already streaming");
            }
        });

    m_webServer.on(
        "/stop_streaming",
        HTTP_GET,
        [this]()
        {
            Serial.println("get: /stop_streaming");
            if (stopUDPStreaming())
            {
                m_webServer.send(200, "text/plain", "OK");
            }
            else
            {
                m_webServer.send(400, "text/plain", "Not streaming");
            }
        });


    m_webServer.on(
        "/file_list",
        HTTP_GET,
        [this]()
        {
            Serial.println("get: /file_list");

            cJSON* root = cJSON_CreateObject();

            // File array
            cJSON* file_array = cJSON_CreateArray();


            File file_root = SD_MMC.open("/");
            file_root.rewindDirectory();
            File current_file = file_root.openNextFile();

            while (current_file)
            {
                // Make shure it is a .dat file
                std::string filename(current_file.name());
                if (filename.find(".dat") < filename.size())
                {
                    // Single file object
                    cJSON* file_object = cJSON_CreateObject();
                    // Filename
                    cJSON_AddStringToObject(file_object, "name", current_file.name());
                    // Size
                    cJSON_AddNumberToObject(file_object, "size", current_file.size());
                    // Add to array
                    cJSON_AddItemToArray(file_array, file_object);
                }

                // Next file
                current_file = file_root.openNextFile();
            }

            // Close file
            file_root.close();
            file_root.rewindDirectory();


            // Adding array to root object
            cJSON_AddItemToObject(root, "files", file_array);
            cJSON_AddStringToObject(root, "download_url", "/file_download");
            cJSON_AddStringToObject(root, "delete_url", "/file_delete");

            String json(cJSON_Print(root));

            // Free memory
            cJSON_free(root);

            Serial.println(json);

            m_webServer.send(200, "application/json", json);
        });


    // URL: /file_delete
    m_webServer.on(
        "/file_delete",
        HTTP_GET,
        [this]()
        {
            if (m_webServer.hasArg("file"))
            {
                String file = "/" + m_webServer.arg("file");
                if (SD_MMC.exists(file))
                {
                    SD_MMC.remove(file);
                    m_webServer.send(200, "text/plain", "OK");
                }
                else
                {
                    m_webServer.send(400, "text/plain", "File not found");
                }
            }
            else
            {
                m_webServer.send(400, "text/plain", "Missing file argument");
            }
        });


    m_webServer.on(
        "/file_download",
        HTTP_GET,
        [this]()
        {
            if (m_webServer.hasArg("file"))
            {
                String file = "/" + m_webServer.arg("file");
                if (SD_MMC.exists(file))
                {
                    File file_to_download = SD_MMC.open(file);
                    // Make sure the receiver treats the file as binary, and gets the correct file size and name
                    m_webServer.sendHeader(
                        "Content-Disposition",
                        String("attachment; filename=") + file_to_download.name());
                    m_webServer.sendHeader("Content-Length", String(file_to_download.size()));
                    m_webServer.streamFile(file_to_download, "application/octet-stream");
                    file_to_download.close();
                    m_webServer.send(200, "text/plain", "OK");
                }
                else
                {
                    m_webServer.send(400, "text/plain", "File not found");
                }
            }
            else
            {
                m_webServer.send(400, "text/plain", "Missing file argument");
            }
        });
}
