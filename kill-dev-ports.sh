#!/bin/bash

echo "🔍 Checking for processes on common development ports..."

# Function to check if a process is recently started (less than 10 seconds old)
is_process_recent() {
    local pid=$1
    if [ -z "$pid" ]; then
        return 1
    fi
    
    # Get process start time in seconds since epoch
    local start_time=$(ps -o lstart= -p $pid 2>/dev/null | xargs -I {} date -d "{}" +%s 2>/dev/null)
    local current_time=$(date +%s)
    
    if [ ! -z "$start_time" ]; then
        local age=$((current_time - start_time))
        if [ $age -lt 10 ]; then
            return 0  # Process is recent (less than 10 seconds old)
        fi
    fi
    return 1  # Process is old or couldn't determine age
}

# Check React dev server (port 3000)
REACT_PID=$(lsof -ti:3000)
if [ ! -z "$REACT_PID" ]; then
    if is_process_recent $REACT_PID; then
        echo "✅ React dev server on port 3000 is recently started, keeping it"
    else
        echo "Found old React dev server on port 3000 (PID: $REACT_PID)"
        kill -9 $REACT_PID
        echo "✅ Killed old React dev server"
    fi
else
    echo "✅ Port 3000 is free"
fi

# Check Flask dev server (port 5000)
FLASK_PID=$(lsof -ti:5000)
if [ ! -z "$FLASK_PID" ]; then
    # Handle multiple PIDs (space or newline separated)
    echo "$FLASK_PID" | while read -r pid; do
        if [ ! -z "$pid" ]; then
            if is_process_recent $pid; then
                echo "✅ Flask dev server on port 5000 (PID: $pid) is recently started, keeping it"
            else
                echo "Found old Flask dev server on port 5000 (PID: $pid)"
                kill -9 $pid 2>/dev/null
                echo "✅ Killed old Flask dev server (PID: $pid)"
            fi
        fi
    done
else
    echo "✅ Port 5000 is free"
fi

# Check for any node processes that might be hanging (but not recently started)
NODE_PIDS=$(pgrep -f "node.*react-scripts")
if [ ! -z "$NODE_PIDS" ]; then
    old_pids=""
    for pid in $NODE_PIDS; do
        if ! is_process_recent $pid; then
            old_pids="$old_pids $pid"
        fi
    done
    
    if [ ! -z "$old_pids" ]; then
        echo "Found old hanging Node.js/React processes:$old_pids"
        echo "$old_pids" | xargs kill -9
        echo "✅ Killed old hanging Node processes"
    else
        echo "✅ All Node processes are recently started, keeping them"
    fi
else
    echo "✅ No hanging Node processes"
fi

echo "🎯 Port cleanup completed! Recent processes preserved." 