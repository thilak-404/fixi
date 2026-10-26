#!/usr/bin/env bash
# ==============================================================================
# FIXI - Rootless Motorola Camera Watchdog
# Streams phone camera via scrcpy window with automatic reconnection
# ==============================================================================

CAM_ID="${1:-0}"

while true; do
    DEVICE="$(adb devices | awk '$2 == "device" { print $1; exit }')"
    if [ -n "$DEVICE" ]; then
        echo "[$(date '+%H:%M:%S')] Launching Motorola Camera (Camera ID: $CAM_ID)..."
        adb -s "$DEVICE" shell "input keyevent KEYCODE_WAKEUP; svc power stayon usb" 2>/dev/null || true
        scrcpy \
            --serial="$DEVICE" \
            --video-source=camera \
            --camera-id="$CAM_ID" \
            --max-fps=15 \
            -m 1280 \
            --stay-awake \
            --no-audio \
            --window-title="Motorola Camera - FIXI Vision" || true
        echo "[$(date '+%H:%M:%S')] Camera disconnected. Reconnecting in 2 seconds..."
    else
        echo "[$(date '+%H:%M:%S')] Waiting for Android device over ADB..."
    fi
    sleep 2
done
