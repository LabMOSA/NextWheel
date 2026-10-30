#ifndef _RTC_H_
#define _RTC_H_

#include <Wire.h>
#include <NextWheel.h>
#include <MCP7940.h>

// POSIX TZ string for America/Toronto (Eastern, DST from 2nd Sunday of March to 1st Sunday of November)
#ifndef NEXTWHEEL_TIMEZONE
#define NEXTWHEEL_TIMEZONE "EST5EDT,M3.2.0,M11.1.0"
#endif

class RTC
{
public:
    RTC();
    void begin();
    void update();
    bool setTime(String time);
    void adjustRTCTime(time_t unix_time);
    void printTime();
    static void setTimeZone();

private:
    MCP7940_Class m_mcp7940;
};

#endif  // _RTC_H_
