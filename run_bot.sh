#!/bin/bash

# Load environment variables from .env file
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Ensure GOOGLE_API_KEY and TZ are set."
fi

# Move to the script's directory (the project root)
cd "$(dirname "$0")"

# Check for virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Error: .venv/bin/activate not found. Did you run ./setup.sh --install?"
    exit 1
fi

start_disk_monitor() {
    if [ -n "$FREQTRADE__TELEGRAM__TOKEN" ] && [ -n "$FREQTRADE__TELEGRAM__CHAT_ID" ] && [ -n "$DISK_USAGE_INTERVAL_MINUTES" ] && [ "$DISK_USAGE_INTERVAL_MINUTES" -gt 0 ] 2>/dev/null; then
        echo "Starting disk usage monitor (every $DISK_USAGE_INTERVAL_MINUTES minutes)..."
        (
            while true; do
                sleep 5 # Wait initial 5s before first check to align with Freqtrade startup
                if ! pgrep -f "freqtrade trade" > /dev/null; then
                    exit 0
                fi

                USED_KB=$(df -k / | awk 'NR==2 {print $3}')
                TOTAL_KB=$(df -k / | awk 'NR==2 {print $2}')
                USED_GB=$(awk "BEGIN {printf \"%.2f\", $USED_KB/1024/1024}")
                TOTAL_GB=$(awk "BEGIN {printf \"%.2f\", $TOTAL_KB/1024/1024}")
                PCT=$(df -k / | awk 'NR==2 {print $5}')

                LOG_DIR="user_data/logs"
                if [ -d "$LOG_DIR" ]; then
                    LOG_SIZE=$(du -sh "$LOG_DIR" 2>/dev/null | awk '{print $1}')
                else
                    LOG_SIZE="0"
                fi

                MSG="💾 Disk Usage: ${USED_GB}GB / ${TOTAL_GB}GB (${PCT})%0A📑 Logs Size: ${LOG_SIZE}"

                curl -s -X POST "https://api.telegram.org/bot${FREQTRADE__TELEGRAM__TOKEN}/sendMessage" \
                    -d chat_id="${FREQTRADE__TELEGRAM__CHAT_ID}" \
                    -d text="$MSG" > /dev/null

                for ((i=0; i<$((DISK_USAGE_INTERVAL_MINUTES * 60)); i++)); do
                    sleep 1
                    if ! pgrep -f "freqtrade trade" > /dev/null; then
                        exit 0
                    fi
                done
            done
        ) &
    fi
}

if [ "$1" == "-r" ] || [ "$1" == "--restart" ]; then
    echo "Restarting Freqtrade background process..."
    pkill -f "freqtrade trade"
    if [ $? -eq 0 ]; then
        echo "Bot stopped successfully. Waiting 10 seconds for cleanup..."
        sleep 10
    else
        echo "No running Freqtrade bot found."
    fi

    # Change the argument to -d so the script falls through to the detached start block below
    set -- "-d"
fi

if [ "$1" == "-k" ] || [ "$1" == "--kill" ] || [ "$1" == "--stop" ]; then
    echo "Stopping Freqtrade background process..."
    pkill -f "freqtrade trade"
    if [ $? -eq 0 ]; then
        echo "Bot stopped successfully."
    else
        echo "No running Freqtrade bot found."
    fi
    exit 0
fi

if [ "$1" == "-d" ] || [ "$1" == "--detached" ]; then
    echo "Starting Freqtrade in the background..."
    start_disk_monitor
    nohup freqtrade trade -v \
        --logfile user_data/logs/freqtrade-spot.log \
        --db-url sqlite:///user_data/tradesv3-spot.sqlite \
        --config user_data/config.json \
        --config user_data/config_spot.json \
        --strategy CustomBestStrategy > user_data/logs/startup.log 2>&1 &

    echo "Bot is running. PID: $!"
    echo "To view logs, run: tail -f user_data/logs/freqtrade-spot.log"
else
    # Run Freqtrade normally
    start_disk_monitor
    freqtrade trade -v \
        --logfile user_data/logs/freqtrade-spot.log \
        --db-url sqlite:///user_data/tradesv3-spot.sqlite \
        --config user_data/config.json \
        --config user_data/config_spot.json \
        --strategy CustomBestStrategy
fi
