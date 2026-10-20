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

## WiFi / WebSocket Protocol

Once a WebSocket connection is established, the client and server exchange data via the WebSocket protocol in binary format : application messages, using the HTTP protocol, are split into one or more frames, adding a little overhead that needs to be considered, especially if we send small amount of data at once.

### WebSocket Binary Message Format

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

> Note 2: To avoid sending small packets with high overhead ratio (data size small vs websocket overhead), the [WebServerTask](arduino_esp32/src/tasks/WebServerTask.h) sends periodic superframes of type 255 at a rate of 20Hz, wich is the aggregation of all accumulated frames in a 50 ms period. This allows to maximize the bandwidth and minimize the TCP/IP embedded stack overhead. Every data frame is timstamped, so we can recover all data chronologically.

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


## REST API

|**Url**   | **HTTP Method** |  **Param(s)**   | **Return** | **Description**   |
|-----|-----------|----|-----------|---|
/config_set_time | POST | *time* (int) = unix timestamp | 200 (OK) |Set system time from unix timestamp
/config_update | POST | accelerometer_precision (int)[], gyrometer_precision (int) [], imu_sampling_rate(int) [], adc_sampling_rate (int)[], encoder_sampling_rate (int)[]| 200 (OK), 400 (Unknown parameter)| Set the sensor configuration
/config | GET | None | 200, JSON={"accelerometer_precision": value, "gyrometer_precision": value, "imu_sampling_rate": value, "adc_sampling_rate": value, "encoder_sampling_rate": value} | Get the current sensor configuration.
/system_state | GET | None | 200, JSON = {"recording": (bool) recording_flag, "streaming": (bool) streaming_flag, "filename": (string) recording_filename } | Return the current state of the system.
/start_recording | GET | None | 200, "OK", 400, "Already recording." | Start recording on SDCard. File name will be automatically generated with current time.
/stop_recording | GET | None | 200, "0K", 400, "Not recording." | Stop recording to SD Card.
/file_list | GET | None | 200, JSON Dict containing {"files": [{"name": file_size(string), "size": file_size(int)}, ...], "download_url": url_for_file_download(string), "delete_url": url_for_file_delete(string)} | Get all the file information stored in the SD Card.
/file_delete | GET | file (string) The name of the file | 200 (OK), 400 (Bad Parameter) | Delete the file from the SD CARD.
/file_download | GET | file (string) The name of the file | The file in binary format or 404 not found. | The file is is transfered from the SDCARD.
