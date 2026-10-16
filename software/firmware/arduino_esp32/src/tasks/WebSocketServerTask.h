#ifndef _WEBSOCKET_SERVER_TASK_H_
#define _WEBSOCKET_SERVER_TASK_H_

#include "config/WiFiConfig.h"
#include "tasks/WorkerTask.h"
#include <NextWheel.h>
#include <data/DataFrame.h>
#include <data/ConfigDataFrame.h>
#include <string.h>
#include <list>
#if defined(_WEBSOCKET_SERVER_ENABLED)
#include <WebSocketsServer.h>
#endif
#include <WebServer.h>
#include "tasks/UDPStreamer.h"


class WebSocketServerTask : public WorkerTask
{
    static const size_t WEBSOCKET_SERVER_STACK_SIZE = 12000;

public:
    WebSocketServerTask();
    virtual void run(void* app) override;
    bool isWebSocketConnected();
    bool isUDPStreaming();


protected:
    
    void setupRESTAPI();
    bool startUDPStreaming();
    bool stopUDPStreaming();

    #ifdef _WEBSOCKET_SERVER_ENABLED
    void webSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length);

    WebSocketsServer m_webSocketServer;
    #endif

    WebServer m_webServer;
    UDPStreamer m_udpStreamer;
};

#endif  // _WEBSOCKET_SERVER_TASK_H_
