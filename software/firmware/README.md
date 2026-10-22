# Nextwheel firmware

This project uses the [PlatformIO](https://platformio.org/) extension for Visual Studio Code for compilation.

## Requirements for building the firmware

* [Zoom Recording for temporary reference (unlisted link)](https://uqam.ca.panopto.com/Panopto/Pages/Viewer.aspx?id=369eabaa-aeb1-4282-85cf-af47011472e2)

* [Visual Studio Code](https://code.visualstudio.com/download) (latest version)
* [PlatformIO extension](https://platformio.org/)
* In VSCode's new PlatformIO menu (left toolbar with alien icon), got to `PIO Home` > `Platforms` and install Espressif 32 Platform @ 5.x.x

## Coding style

* Use the clang-format Visual Studio Extension to follow coding style defined in the [.clang-format](.clang-format) file.

## Programming

> J14 and J15 jumpers must be removed before programming. This is required for the UART0 to work.
> It is recommended that you leave J14 and J15 jumpers unconnected to get the console output (serial monitor).

Steps:

1. Copy [lib/NextWheel/src/config/WiFiConfig.template.h](lib/NextWheel/src/config/WiFiConfig.template.h) as [lib/NextWheel/src/config/WiFiConfig.h](lib/NextWheel/src/config/WiFiConfig.h) and modify it according to your WiFi information.
2. Go to the PlatformIO menu (left toolbar with alien icon)
3. Click on *General/Build*
4. Click on *General/Upload*

> Note from version 0.2, we are not serving web pages anymore and the data section of the flash is unused.

## Running

> Make sure J11 jumper is in the 1-2 position. This will enable sound from the ESP32.

## Hardware

The following sections will describe the supported hardware / software modules.

### I2C

IMU: SEN0373 (BMX160 9-AXIS SENSOR MODULE)
Default address : 0x68

RTC: MCP7940M (Real time clock with battery backup)
Default address : 0xDE

POWER: INA220
Default address : 0x40

### SPI

ADS8688  IDBTR

## WiFi / UDP Streaming Protocol

Sensor data is streamed over UDP by the [UDPStreamer](arduino_esp32/src/tasks/UDPStreamer.cpp), while control is done with the [REST API](#rest-api) over HTTP (port 80).

1. The client opens a UDP socket bound to a port of its choice (Python API default: 12345).
2. The client calls `/start_streaming?address=x.x.x.x&port=n` to give the destination of the stream. The address can be the IP of the client (unicast, recommended) or a broadcast address such as `255.255.255.255` (default of the Python API), which reaches every host on the local network.
3. The wheel first sends a CONFIG frame (type 1), required to convert raw values, then data superframes (type 255). A new CONFIG frame is sent every time the configuration changes (`/config_update`). Data frames still queued with the previous configuration are discarded.
4. The client calls `/stop_streaming` to stop the stream.

Since UDP is connectionless:

* There is no retransmission and no ordering guarantee. Datagrams are sent non-blocking and are dropped (counted as dropped packets) if the WiFi stack is out of buffers. Clients must tolerate lost datagrams.
* Each datagram is self-contained: it holds one or more complete frames, never a partial frame, so it can be decoded independently of the others.
* Streaming continues until `/stop_streaming` is called, even if the client goes away.

### UDP Binary Message Format

> Note 1: Bytes are ordered little endian (lsb first)
> Note 2: Protocol optimized for more efficiency using raw data when possible. Conversions need to occur in client code.

| TYPE (uint8)   | TIMESTAMP (uint64)   | DATA SIZE (uint8)   | DATA (variable byte(s)) |
|----------------|----------------------|---------------------|-------------------------|
| 0=UNKNOWN      | INVALID              | 0 BYTE              | NONE                    |
| 1=CONFIG       | UNIX MICROSECONDS    | 20 BYTES            | 5x UINT32 (See [ConfigData](lib/NextWheel/src/config/GlobalConfig.h)) |
| 2=ADC          | UNIX MICROSECONDS    | 16 BYTES            | 8 CH x UINT16 (See [ADCDataFrame](lib/NextWheel/src/data/ADCDataFrame.h)) |
| 3=IMU          | UNIX MICROSECONDS    | 36 BYTES            | 9 INT16 (AX,AY,AZ, GX,GY,GZ, MX,MY,MZ) (See [IMUDataFrame](lib/NextWheel/src/data/IMUDataFrame.h)) |
| 4=POWER        | UNIX MICROSECONDS    | 13 BYTES            | 3 FLOAT32 (V,I,P) + FLAGS (uint8) (See [PowerDataFrame](lib/NextWheel/src/data/PowerDataFrame.h)) |
| 5=RTC          | UNIX MICROSECONDS    | NOT IMPLEMENTED YET | NONE                    |
| 6=AUDIO        | UNIX MICROSECONDS    | NOT IMPLEMENTED YET | NONE                    |
| 7=QUADENCODER  | UNIX MICROSECONDS    | 8 BYTES             | 1 INT64 (See [QuadEncoderDataFrame](lib/NextWheel/src/data/QuadEncoderDataFrame.h))|
| 255=SUPERFRAME | UNIX MICROSECONDS    | NB INCLUDED FRAMES  | VARIABLE NUMBER OF FULL FRAMES |

> Note 3: To avoid sending small packets with a high overhead ratio (small data size vs UDP/IP headers), the [WebServerTask](arduino_esp32/src/tasks/WebServerTask.cpp) aggregates all queued frames into a superframe of type 255, sent as a single UDP datagram. A superframe is sent when its payload reaches approximately 1400 bytes (to stay below the WiFi MTU and avoid IP fragmentation) or when no new frame arrives for 50 ms. Every data frame is timestamped, so all data can be recovered chronologically.

### ADC and IMU VALUE CONVERSION

```python
class GlobalConfig:
    """
    Global configuration of the instrumented wheel. This will be used to store current configuration
    and calculate conversions from raw data.
    """
    def __init__(self):
        # IMU CONFIG
        self.accel_range = 16
        self.gyro_range = 2000
        self.mag_range = 2500
        self.imu_rate = 240

        # ADC CONFIG
        self.adc_rate = 1000

        # ENCODER CONFIG
        self.encoder_rate = 240

        # ACCORDING TO ADS8688 DATASHEET
        self._adc_v_ref = 4.096
        # ADC RANGING from [-5,5] V according to PCB design
        self._adc_in_max = 1.25 * self._adc_v_ref
        self._adc_in_min = -1.25 * self._adc_v_ref

        # ACCORDING TO BMX160 DATASHEET
        self._gravity_earth = 9.80665
        self._accel_mg_lsb_2g = 0.000061035
        self._accel_mg_lsb_4g = 0.000122070
        self._accel_mg_lsb_8g = 0.000244141
        self._accel_mg_lsb_16g = 0.000488281
        self._gyro_sensitivity_125dps = 0.0038110
        self._gyro_sensitivity_250dps = 0.0076220
        self._gyro_sensitivity_500dps = 0.0152439
        self._gyro_sensitivity_1000dps = 0.0304878
        self._gyro_sensitivity_2000dps = 0.0609756
        self._mag_ut_lsb = 0.3

    def update_config(self, accel_range: int, gyro_range: int, mag_range: int,  imu_rate: int, adc_rate: int, encoder_rate: int):
        self.accel_range = accel_range
        self.gyro_range = gyro_range
        self.mag_range = mag_range
        self.imu_rate = imu_rate
        self.adc_rate = adc_rate
        self.encoder_rate = encoder_rate

    def convert_adc_values(self, values: Tuple[int]) -> Tuple[float]:
        return tuple(self.convert_adc_value(value) for value in values)

    def convert_adc_value(self, value: int) -> float:
        # This is according to the 86888 datasheet
        return float(value) * (self._adc_in_max - self._adc_in_min) / 65535. + self._adc_in_min

    def convert_accel_values(self, values: Tuple[int]) -> Tuple[float]:
        return tuple(self.convert_accel_value(value) for value in values)

    def convert_accel_value(self, values: int) -> float:
        if self.accel_range == 2:
            return float(values) * self._accel_mg_lsb_2g * self._gravity_earth
        elif self.accel_range == 4:
            return float(values) * self._accel_mg_lsb_4g * self._gravity_earth
        elif self.accel_range == 8:
            return float(values) * self._accel_mg_lsb_8g * self._gravity_earth
        elif self.accel_range == 16:
            return float(values) * self._accel_mg_lsb_16g * self._gravity_earth
        else:
            print('Invalid accel range')
            return 0

    def convert_gyro_values(self, values: Tuple[int]) -> Tuple[float]:
        return tuple(self.convert_gyro_value(value) for value in values)

    def convert_gyro_value(self, values: int) -> float:
        if self.gyro_range == 125:
            return float(values) * self._gyro_sensitivity_125dps
        elif self.gyro_range == 250:
            return float(values) * self._gyro_sensitivity_250dps
        elif self.gyro_range == 500:
            return float(values) * self._gyro_sensitivity_500dps
        elif self.gyro_range == 1000:
            return float(values) * self._gyro_sensitivity_1000dps
        elif self.gyro_range == 2000:
            return float(values) * self._gyro_sensitivity_2000dps
        else:
            print('Invalid gyro range')
            return 0

    def convert_mag_values(self, values: Tuple[int]) -> Tuple[float]:
        return tuple(self.convert_mag_value(value) for value in values)

    def convert_mag_value(self, values: int) -> float:
        if self.mag_range == 2500:
            return float(values) * self._mag_ut_lsb
        else:
            print('Invalid mag range')
            return 0

```

### POWER VALUE CONVERSION

Values are already converted on device. Size optimization of the binary stream was not very practical and useful for those low bandwidth values.


## Time Synchronization (NTP)

The system time is used to timestamp every data frame (UNIX microseconds) and to generate recording file names.

* At boot, the system time is initialized from the [RTC](arduino_esp32/lib/NextWheel/src/RTC.cpp) (MCP7940M with battery backup). The RTC stores UTC time.
* When an IP address is obtained at startup, SNTP is started by the [WebServerTask](arduino_esp32/src/tasks/WebServerTask.cpp). If the connection is lost before the first synchronization, SNTP is restarted on reconnection.
* After the first synchronization, SNTP is stopped. There is no periodic resynchronization, so the system time drifts until the next boot or until a resynchronization is requested with `/ntp_resync`. SNTP is stopped again after that synchronization.
* Corrections are applied gradually (`SNTP_SYNC_MODE_SMOOTH`, using `adjtime`) to avoid timestamp jumps while recording or streaming. If the error is larger than 35 minutes (e.g. RTC never set), the time is set immediately.
* After each synchronization, the NTP time is written back to the RTC so the next boot starts with a corrected time, even without WiFi.
* The NTP server requires Internet access from the WiFi network. Without it, the system keeps running on RTC time.
* `/config_set_time` still sets the time manually, but it will be overridden by the startup NTP synchronization or by `/ntp_resync`.

| **Define** | **Default** | **Description** |
|------------|-------------|-----------------|
| `NTP_DEFAULT_SERVER` | `"pool.ntp.org"` | NTP server ([WebServerTask.h](arduino_esp32/src/tasks/WebServerTask.h)) |
| `NEXTWHEEL_TIMEZONE` | `"EST5EDT,M3.2.0,M11.1.0"` | POSIX TZ string for local time, America/Toronto ([RTC.h](arduino_esp32/lib/NextWheel/src/RTC.h)) |

Both can be overridden with `build_flags` in [platformio.ini](arduino_esp32/platformio.ini), for example:

```ini
build_flags = -DNTP_DEFAULT_SERVER=\"ca.pool.ntp.org\"
```

## REST API

All endpoints are served on HTTP port 80. Text responses are `text/plain`, JSON responses are `application/json`. Unknown URLs return 404 with the request details.

|**Url**   | **HTTP Method** |  **Param(s)**   | **Return** | **Description**   |
|-----|-----------|----|-----------|---|
/ | GET | None | 200, "Hello from NextWheel REST API Server !" | Check that the server is running.
/config_set_time | POST | time (int) unix timestamp in seconds | 200 (OK), 400 (Missing time argument) | Set the system time and the RTC from a unix timestamp. Overridden by the startup NTP synchronization or `/ntp_resync`, see [Time Synchronization](#time-synchronization-ntp).
/ntp_time | GET | None | 200, JSON = {"synced": (bool) ntp_response_received, "server": (string) ntp_server, "time": (float) current_unix_time_seconds, "local_time": (string) ISO 8601 local time with UTC offset, "last_sync": (int) unix_time_of_last_sync, "adjusting": (float) remaining_gradual_correction_seconds} | Get the current time and NTP synchronization status. `adjusting` is 0 when the system time is fully synchronized.
/ntp_resync | GET | None | 200 (OK), 400 (Not connected) | Restart SNTP for a single synchronization. It is stopped again once the time is received and written to the RTC. Check `last_sync` in `/ntp_time` to know when it is done.
/config_update | POST | accelerometer_precision (int), gyrometer_precision (int), imu_sampling_rate (int), adc_sampling_rate (int), encoder_sampling_rate (int). All are required. | 200 (OK), 400 (Missing arguments) | Set and save the sensor configuration. It is applied immediately to all sensors, and a new CONFIG frame is written to the current recording and sent on the UDP stream.
/config | GET | None | 200, JSON = {"accelerometer_precision": (int), "gyrometer_precision": (int), "imu_sampling_rate": (int), "adc_sampling_rate": (int), "encoder_sampling_rate": (int)} | Get the current sensor configuration.
/system_state | GET | None | 200, JSON = {"recording": (int 0/1) recording_flag, "streaming": (int 0/1) streaming_flag, "filename": (string) recording_filename} | Get the current state of the system.
/start_streaming | GET | address (string) destination IP (x.x.x.x, unicast or broadcast), port (int) destination UDP port (1-65534) | 200 (OK), 400 (Missing arguments, Invalid address, Invalid port, Already streaming) | Start streaming data over UDP to address:port. See [UDP Streaming Protocol](#wifi--udp-streaming-protocol).
/stop_streaming | GET | None | 200 (OK), 400 (Not streaming) | Stop UDP streaming.
/start_recording | GET | None | 200 (OK), 400 (Already recording) | Start recording on the SD card. The file name is generated from the current local time (`log_YYYY-MM-DD_HH-MM-SS.dat`).
/stop_recording | GET | None | 200 (OK), 400 (Not recording) | Stop recording on the SD card.
/file_list | GET | None | 200, JSON = {"files": [{"name": (string) file_name, "size": (int) file_size_bytes}, ...], "download_url": "/file_download", "delete_url": "/file_delete"} | List the recording files (`.dat`) stored on the SD card.
/file_delete | GET | file (string) file name | 200 (OK), 400 (File not found, Missing file argument) | Delete a file from the SD card.
/file_download | GET | file (string) file name | 200, file content (`application/octet-stream`, with `Content-Disposition` and `Content-Length`), 400 (File not found, Missing file argument) | Download a file from the SD card.
