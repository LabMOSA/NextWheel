#ifndef _WEB_SERVER_TASK_H_
#define _WEB_SERVER_TASK_H_

#include "config/WiFiConfig.h"
#include "tasks/WorkerTask.h"
#include <NextWheel.h>
#include <data/DataFrame.h>
#include <data/ConfigDataFrame.h>
#include <string.h>
#include <list>
#include <WebServer.h>
#include "tasks/UDPStreamer.h"


class WebServerTask : public WorkerTask
{
    static const size_t WEB_SERVER_STACK_SIZE = 12000;

public:
    WebServerTask();
    virtual void run(void* app) override;
    bool isUDPStreaming();


protected:
    
    void setupRESTAPI();
    bool startUDPStreaming(uint32_t address, uint16_t port);
    bool stopUDPStreaming();
    WebServer m_webServer;
    UDPStreamer m_udpStreamer;
};

#endif  // _WEB_SERVER_TASK_H_
