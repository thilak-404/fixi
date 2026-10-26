#!/usr/bin/env bash
# FIXI - Camera Stream Watchdog for scrcpy -> /dev/video0
# Keeps phone camera streaming continuously with auto-restart on disconnect

CAM_ID="${1:-0}"
echo "=========================================================="
echo " Starting FIXI Camera Stream to /dev/video0 (Camera ID: $CAM_ID)"
echo "=========================================================="

if ! command -v adb >/dev/null 2>&1 || ! command -v scrcpy >/dev/null 2>&1; then
    echo "adb and scrcpy are required for the phone camera bridge."
    exit 0
fi

while [ ! -e /dev/video0 ]; do
    echo "[$(date '+%H:%M:%S')] /dev/video0 is not created yet. Waiting for 'sudo modprobe v4l2loopback'..."
    sleep 3
done

while true; do
    DEVICE="$(adb devices | awk '$2 == "device" { print $1; exit }')"
    if [ -z "$DEVICE" ]; then
        echo "[$(date '+%H:%M:%S')] No authorized Android device. Waiting..."
        sleep 2
        continue
    fi

    # Ensure phone stays awake
    adb -s "$DEVICE" shell svc power stayon usb 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] Launching scrcpy camera stream (Camera $CAM_ID)..."
    scrcpy \
        --serial="$DEVICE" \
        --video-source=camera \
        --camera-id="$CAM_ID" \
        -m 640 \
        --max-fps=15 \
        --stay-awake \
        --no-audio \
        --v4l2-sink=/dev/video0 || true

    echo "[$(date '+%H:%M:%S')] Camera stream ended or disconnected. Reconnecting in 2s..."
    sleep 2
done
