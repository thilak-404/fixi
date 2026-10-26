#!/usr/bin/env bash
# ==============================================================================
# Fixi - Unified System Launcher
# Starts 100% Local AI Person-Aware Companion (CV + Robotics + Ollama + Web UI)
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=================================================="
echo "  Starting FIXI Local AI Companion Platform"
echo "=================================================="

# 1. Virtual Webcam Module (v4l2loopback)
if [ ! -e /dev/video0 ]; then
    echo "[1/4] Setting up /dev/video0 loopback device..."
    pkexec modprobe v4l2loopback devices=1 video_nr=0 card_label="Virtual Webcam" exclusive_caps=1 || true
else
    echo "[1/4] /dev/video0 is ready."
fi

# 2. Local Ollama LLM Service
OLLAMA_BIN="${OLLAMA_BIN:-/home/regnad/.local/bin/ollama}"
if ! curl -fsS http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "[2/4] Starting Ollama LLM service..."
    if [ -x "$OLLAMA_BIN" ]; then
        nohup "$OLLAMA_BIN" serve > /tmp/fixi_ollama.log 2>&1 &
        for _ in $(seq 1 10); do
            curl -fsS http://127.0.0.1:11434/api/tags > /dev/null 2>&1 && break
            sleep 1
        done
    else
        echo "[2/4] Ollama binary not found at $OLLAMA_BIN; local LLM remains offline."
    fi
else
    echo "[2/4] Ollama LLM service is already active."
fi

# 3. Android Phone Camera Stream (via scrcpy -> /dev/video0)
if adb devices | grep -q "device$"; then
    echo "[3/4] Android camera device detected. Starting camera stream..."
    adb shell "input keyevent KEYCODE_WAKEUP; am force-stop com.motorola.camera3 2>/dev/null || true"
    if pgrep -af "start_camera_stream.sh" >/dev/null 2>&1; then
        echo "[3/4] Camera stream watchdog is already running."
    else
        nohup ./scripts/start_camera_stream.sh 0 > /tmp/fixi_camera.log 2>&1 &
    fi
    sleep 2
else
    echo "[3/4] Notice: No physical phone detected over ADB. Using fallback/synthetic camera."
fi

# 4. FastAPI Backend + Robotics Simulation + Web UI
echo "[4/4] Starting FIXI Backend & Dashboard on http://localhost:8000 ..."
"$PROJECT_DIR/.venv/bin/uvicorn" dashboard.app:app --host 0.0.0.0 --port 8000
