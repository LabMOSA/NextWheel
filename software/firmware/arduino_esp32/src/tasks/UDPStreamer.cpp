#include "UDPStreamer.h"
#include <config/GlobalConfig.h>
#include <lwip/sockets.h>


UDPStreamer::UDPStreamer() : m_socket(-1), m_streaming(false), m_droppedPackets(0) {}

UDPStreamer::~UDPStreamer()
{
    stop();
}

bool UDPStreamer::start()
{
    if (m_socket < 0)
    {
        m_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (m_socket < 0)
        {
            Serial.printf("UDPStreamer::start: socket() failed, errno: %i\n", errno);
            return false;
        }

        // Allow sending to broadcast addresses (default is 255.255.255.255)
        int broadcast = 1;
        setsockopt(m_socket, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));
    }

    m_droppedPackets = 0;
    m_streaming = true;
    return true;
}

void UDPStreamer::stop()
{
    m_streaming = false;
    if (m_socket >= 0)
    {
        close(m_socket);
        m_socket = -1;
    }
}

bool UDPStreamer::isStreaming() const
{
    return m_streaming;
}

bool UDPStreamer::send(const uint8_t* data, size_t size)
{
    if (!m_streaming || m_socket < 0)
    {
        return false;
    }

    // Address is stored in network byte order (IPAddress raw value)
    struct sockaddr_in dest = {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(GlobalConfig::instance().get_udp_broadcast_port());
    dest.sin_addr.s_addr = GlobalConfig::instance().get_udp_broadcast_address();

    // Non blocking, if the WiFi stack is out of buffers, drop the packet
    int sent = sendto(m_socket, data, size, MSG_DONTWAIT, (struct sockaddr*)&dest, sizeof(dest));
    if (sent != (int)size)
    {
        m_droppedPackets++;
        return false;
    }
    return true;
}

bool UDPStreamer::sendFrame(const DataFrame& frame)
{
    uint8_t buffer[frame.getTotalSize()];
    frame.serialize(buffer, frame.getTotalSize());
    return send(buffer, frame.getTotalSize());
}

uint32_t UDPStreamer::getDroppedPackets() const
{
    return m_droppedPackets;
}
