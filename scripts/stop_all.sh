#!/usr/bin/env bash
# ==============================================================================
# Fixi - Stop All Services
# ==============================================================================

echo "Stopping FIXI services..."
pkill -f "uvicorn dashboard.app:app" 2>/dev/null || true
pkill -f "scripts/start_camera_stream.sh" 2>/dev/null || true
pkill -f "scrcpy" 2>/dev/null || true
pkill -f "ollama serve" 2>/dev/null || true
echo "All FIXI services stopped."
