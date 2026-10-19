#ifndef _POWER_SENSOR_TASK_H_
#define _POWER_SENSOR_TASK_H_


#include "tasks/SensorTask.h"
#include "data/PowerDataFrame.h"
#include "Power.h"


class PowerSensorTask : public SensorTask
{
    // Larger than default, Serial.printf with %f needs a lot of stack
    static const size_t POWER_SENSOR_TASK_STACK_SIZE = 4096;

public:
    PowerSensorTask();
    virtual void run(void* app) override;

private:
    Power m_power;
};

#endif  // _POWER_SENSOR_TASK_H_
