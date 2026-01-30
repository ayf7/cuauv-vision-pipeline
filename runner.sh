#!/usr/bin/env bash

# utility to watch a vision module for changes, reloading if a change was
# detected

# autocomplete under install/zsh_autocomplete_vision_runner_complete.zsh

PYTHON_PID=""
INOTIFY_PID=""

print_help() {
    # all arguments are forwarded to the vision module
    echo "usage: auv-vision-runner VISION_MODULE [vision module args...]"
    exit 0
}

if [ "$#" -lt 1 ]; then
    print_help
fi

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    print_help
fi

stop_python_process() {
    # if python pid exists and the process is still alive
    if [[ -n "$PYTHON_PID" ]] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        stdbuf -o0 echo "[auv-vr]: terminating previous Python process: (PID: ${PYTHON_PID})."
        kill -s INT "$PYTHON_PID"
        wait "$PYTHON_PID"
        PYTHON_PID=""
        return 0
    fi
}

stop_inotify_process() {
    if [[ -n "$INOTIFY_PID" ]] && kill -0 "$INOTIFY_PID" 2>/dev/null; then
        stdbuf -o0 echo "[auv-vr]: terminating inotify process: (PID: ${INOTIFY_PID})."
        kill "$INOTIFY_PID"
        wait "$INOTIFY_PID"
        INOTIFY_PID=""
        return 0
    fi
}

cleanup() {
    stdbuf -o0 echo "[auv-vr]: Caught Ctrl-C, terminating vision module."
    stop_python_process
    stop_inotify_process
    exit 0
}

# shellcheck disable=SC2154
# why: software_path is a global variable defined in install/zshrc
SCRIPT_PATH="${software_path}/vision/modules/$1.py"
shift 1

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Error: The script '$SCRIPT_PATH' does not exist."
    exit 1
fi

# call stop_python_process if receive any of the following signals
trap cleanup TERM INT

echo "[auv-vr]: Monitoring '$SCRIPT_PATH' for changes."
echo "[auv-vr]: Press Ctrl+C to stop."
(python3 "${SCRIPT_PATH}" "$@") &
PYTHON_PID=$!

sleep 0.5
if ! kill -0 "$PYTHON_PID" 2>/dev/null; then
    echo "[auv-vr]: Python script terminated immediately. Will not watch!"
    wait "$PYTHON_PID" 2>/dev/null || true
    exit "$?"
fi

while true; do
    # capture inotifywait pid to kill when this script exits
    (inotifywait -e modify "$SCRIPT_PATH" &>/dev/null) &
    INOTIFY_PID=$!
    wait $INOTIFY_PID

    echo ""
    echo ""
    echo "[auv-vr]: Detected vision module change."
    stop_python_process
    echo "[auv-vr]: Starting new Python process for $SCRIPT_PATH."
    (python3 "${SCRIPT_PATH}" "$@") &
    PYTHON_PID=$!
    sleep 1
done
