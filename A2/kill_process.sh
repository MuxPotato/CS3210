#!/bin/bash

# Get the current SSH session PID so it doesn't get killed
ssh_pid=$(ps -o pid= -p $$)

# Find and list all processes owned by your user except the SSH session and essential system processes
processes_to_kill=$(ps -u $USER -o pid,comm | grep -vE "(sshd|bash|ps|grep|$$|$ssh_pid)" | awk '{print $1}')

# Loop over the process IDs and kill them
if [ -z "$processes_to_kill" ]; then
    echo "No processes to kill."
else
    echo "Killing the following processes:"
    echo "$processes_to_kill"
    kill -9 $processes_to_kill
fi
