#ifndef _UDP_STREAMER_H_
#define _UDP_STREAMER_H_

#include <Arduino.h>
#include <data/DataFrame.h>

/**
 * Sends binary frames (see "UDP Binary Message Format" in README) to the UDP
 * address / port stored in GlobalConfig. The destination is read from
 * GlobalConfig on every send so configuration changes apply immediately.
 */
class UDPStreamer
{
public:
    UDPStreamer();
    ~UDPStreamer();

    bool start();
    void stop();
    bool isStreaming() const;

    bool send(const uint8_t* data, size_t size);
    bool sendFrame(const DataFrame& frame);

    uint32_t getDroppedPackets() const;

private:
    int m_socket;
    bool m_streaming;
    uint32_t m_droppedPackets;
};

#endif  // _UDP_STREAMER_H_
