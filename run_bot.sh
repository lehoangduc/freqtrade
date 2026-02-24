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

if [ "$1" == "-d" ] || [ "$1" == "--detached" ]; then
    echo "Starting Freqtrade in the background..."
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
    freqtrade trade -v \
        --logfile user_data/logs/freqtrade-spot.log \
        --db-url sqlite:///user_data/tradesv3-spot.sqlite \
        --config user_data/config.json \
        --config user_data/config_spot.json \
        --strategy CustomBestStrategy
fi
