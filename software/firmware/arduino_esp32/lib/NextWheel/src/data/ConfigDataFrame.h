#ifndef _CONFIG_DATA_FRAME_H_
#define _CONFIG_DATA_FRAME_H_

#include <data/DataFrame.h>
#include <config/GlobalConfig.h>
#include <cstddef>

class ConfigDataFrame : public DataFrame
{
public:
    // Only the sensor configuration is serialized (network settings are excluded)
    // to keep the frame format compatible with existing clients and recordings.
    const static size_t CONFIG_DATA_FRAME_SIZE = offsetof(GlobalConfig::ConfigData, udp_broadcast_address);

    ConfigDataFrame(const GlobalConfig::ConfigData& configData, uint64_t timestamp = DataFrame::getCurrentTimeStamp());

    ConfigDataFrame(const ConfigDataFrame& other);

    virtual size_t serializePayload(uint8_t* buffer, size_t buffer_size) const override;

    virtual DataFrame* clone() const override;

protected:
    GlobalConfig::ConfigData m_configData;
};

#endif  // _CONFIG_DATA_FRAME_H_
