#ifndef _GLOBAL_CONFIG_H_
#define _GLOBAL_CONFIG_H_

#include "NextWheel.h"

#include <Arduino.h>
#include <EEPROM.h>
#include <vector>

class GlobalConfig
{
public:
    typedef struct
    {
        uint32_t accel_range;
        uint32_t gyro_range;
        uint32_t mag_range;
        uint32_t imu_sample_rate;
        uint32_t adc_sample_rate;
        uint32_t encoder_sample_rate;
        uint32_t udp_broadcast_address;
        uint16_t udp_broadcast_port;
    } ConfigData;


    void begin();

    void load();

    void print();

    void save();

    ConfigData get() const;

    void set(ConfigData data);

    bool validate();

    void set_accel_range(unsigned int range);

    uint32_t get_accel_range() const;

    void set_gyro_range(unsigned int range);

    uint32_t get_gyro_range() const;

    void set_mag_range(unsigned int range);

    uint32_t get_mag_range() const;

    void set_imu_sample_rate(unsigned int rate);

    uint32_t get_imu_sample_rate() const;

    void set_adc_sample_rate(unsigned int rate);

    uint32_t get_adc_sample_rate() const;

    void set_encoder_sample_rate(unsigned int rate);

    uint32_t get_encoder_sample_rate() const;

    void set_udp_broadcast_address(uint32_t address);
    
    uint32_t get_udp_broadcast_address() const;
    
    void set_udp_broadcast_port(uint16_t port);
    
    uint16_t get_udp_broadcast_port() const;

    void setDefault();

    // Singleton instance
    static GlobalConfig& instance();

private:
    ConfigData m_config;
    GlobalConfig() = default;
    ~GlobalConfig() = default;

    // Vectors containing valid configuration values
    static const std::vector<uint32_t> m_accel_ranges;
    static const std::vector<uint32_t> m_gyro_ranges;
    static const std::vector<uint32_t> m_mag_ranges;
    static const std::vector<uint32_t> m_imu_sampling_rate_ranges;
    static const std::vector<uint32_t> m_adc_sampling_rate_ranges;
    static const std::vector<uint32_t> m_encoder_sampling_rate_ranges;
};

#endif  // _GLOBAL_CONFIG_H_
