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

#ifndef NTP_DEFAULT_SERVER
#define NTP_DEFAULT_SERVER "pool.ntp.org"
#endif

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
    void setupNTP();
    void startNTP();
    static void onNTPTimeSync(struct timeval* tv);

    // Written from the SNTP callback (lwIP task), read from this task
    static volatile bool s_ntpSynced;
    static volatile bool s_ntpRTCUpdatePending;
    static volatile time_t s_ntpLastSyncTime;
    static volatile int64_t s_ntpLastSyncTimerUs;

    WebServer m_webServer;
    UDPStreamer m_udpStreamer;
};

#endif  // _WEB_SERVER_TASK_H_
