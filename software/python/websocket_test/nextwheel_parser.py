import datetime
import struct


def parse_power_frame(message: bytes):
    if len(message) != 13:
        return []
    else:
        vals = struct.unpack_from("<3fB", message)
        return vals


def parse_imu_frame(message: bytes):
    if len(message) != 36:
        return []
    else:
        vals = struct.unpack_from("<9f", message)
        return vals


def parse_adc_frame(message: bytes):
    if len(message) != 32:
        return []
    else:
        vals = struct.unpack_from("<8f", message)
        return vals


def parse_config_frame(message: bytes):
    if len(message) != 20:
        return []
    else:
        vals = struct.unpack_from("<5I", message)
        print(
            f"Config accel_range:{vals[0]}, gyro_range:{vals[1]}, "
            f"mag_range:{vals[2]}, imu_sample_rate:{vals[3]}, adc_sample_rate:{vals[4]}"
        )
        return vals


def parse_encoder_frame(message: bytes):
    if len(message) != 8:
        return []
    else:
        vals = struct.unpack_from("<q", message)
        return vals


def parse_superframe(message: bytes, count: int):
    offset = 0
    header_size = 10

    adc_values = []
    imu_values = []
    power_values = []
    encoder_values = []

    for sub_count in range(count):
        (frame_type, timestamp, data_size) = struct.unpack_from(
            "<BQB", message[offset : offset + header_size]
        )
        # print(f'sub header: {sub_count}/{count}', frame_type, timestamp, data_size)

        # Convert to real time
        timestamp = datetime.datetime.fromtimestamp(timestamp / 1e6)

        if frame_type == 2:
            adc_values.append(
                (
                    timestamp,
                    parse_adc_frame(
                        message[
                            offset + header_size : offset
                            + header_size
                            + data_size
                        ]
                    ),
                )
            )
        elif frame_type == 3:
            imu_values.append(
                (
                    timestamp,
                    parse_imu_frame(
                        message[
                            offset + header_size : offset
                            + header_size
                            + data_size
                        ]
                    ),
                )
            )
        elif frame_type == 4:
            power_values.append(
                (
                    timestamp,
                    parse_power_frame(
                        message[
                            offset + header_size : offset
                            + header_size
                            + data_size
                        ]
                    ),
                )
            )
        elif frame_type == 7:
            encoder_values.append(
                (
                    timestamp,
                    parse_encoder_frame(
                        message[
                            offset + header_size : offset
                            + header_size
                            + data_size
                        ]
                    ),
                )
            )

        offset = offset + data_size + header_size

    return {
        "adc_values": adc_values,
        "imu_values": imu_values,
        "power_values": power_values,
        "encoder_values": encoder_values,
    }


def parse(message: bytes):
    # uint8 type, uint64 timestamp, uint8 datasize (little endian)
    # print('message', len(message), message[0:10].hex())

    offset = 0
    header_size = 10

    adc_values = []
    imu_values = []
    power_values = []
    encoder_values = []
    config = []

    while offset <= len(message):
        if offset + header_size >= len(message):
            break

        (frame_type, timestamp, data_size) = struct.unpack_from(
            "<BQB", message[offset : offset + header_size]
        )

        # Convert to real time
        timestamp = datetime.datetime.fromtimestamp(timestamp / 1e6)

        # Config frame (should always be first)
        if frame_type == 1:
            print("ConfigFrame detected")
            config.append(
                parse_config_frame(
                    message[
                        offset + header_size : offset + header_size + data_size
                    ]
                )
            )

        if frame_type == 2:
            adc_values.append(
                (
                    timestamp,
                    parse_adc_frame(
                        message[
                            offset + header_size : offset
                            + header_size
                            + data_size
                        ]
                    ),
                )
            )
        elif frame_type == 3:
            imu_values.append(
                (
                    timestamp,
                    parse_imu_frame(
                        message[
                            offset + header_size : offset
                            + header_size
                            + data_size
                        ]
                    ),
                )
            )
        elif frame_type == 4:
            power_values.append(
                (
                    timestamp,
                    parse_power_frame(
                        message[
                            offset + header_size : offset
                            + header_size
                            + data_size
                        ]
                    ),
                )
            )
        elif frame_type == 7:
            encoder_values.append(
                (
                    timestamp,
                    parse_encoder_frame(
                        message[
                            offset + header_size : offset
                            + header_size
                            + data_size
                        ]
                    ),
                )
            )

        offset = offset + data_size + header_size

        print(offset, len(message))

    return {
        "config": config,
        "adc_values": adc_values,
        "imu_values": imu_values,
        "power_values": power_values,
        "encoder_values": encoder_values,
    }
