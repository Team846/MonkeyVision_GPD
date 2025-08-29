#!/bin/bash

source /home/orangepi/MonkeyVision_GPD/gpd_env/bin/activate
cd /home/orangepi/MonkeyVision_GPD/

MAX_CPU=99 
MAX_MEM=80 

check_usage() {
    CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}') 
    MEM=$(free | awk '/Mem:/ {print $3/$2 * 100.0}')   

    CPU=${CPU%.*}
    MEM=${MEM%.*}

    if [ "$CPU" -gt "$MAX_CPU" ] || [ "$MEM" -gt "$MAX_MEM" ]; then
        echo "High resource usage detected: CPU=${CPU}%, MEM=${MEM}%"
        return 1
    else
        return 0
    fi
}

while true; do
    python3 main.py --pipeline 1 &
    PID=$!

    while kill -0 $PID 2>/dev/null; do
        sleep 5 
        check_usage
        if [ $? -ne 0 ]; then
            echo "Restarting due to high usage..."
            kill -9 $PID
            sleep 2
            break
        fi
    done

    sleep 2
done
