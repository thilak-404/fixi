"""
FastAPI Dashboard Server for Fixi.
Serves real-time system monitoring, camera streaming, user management,
memory inspection/editing, and interactive dialogue simulator.
100% Local.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import io
import time
import base64
import tempfile
import urllib.parse
import asyncio
import cv2
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Form, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.config.settings import settings
from app.events import event_bus, EventType
from app.main import fixi_engine, FixiState
from app.users.user_manager import user_manager
from app.users.consent import ConsentStatus
from app.memory.memory_manager import memory_manager
from app.speech.tts import tts_engine
from app.speech.stt import stt_engine
from app.robotics.controller import robot_controller
from app.robotics.simulation import robotics_sim
from app.robotics.benchmark import run_robotics_benchmarks
from app.memory.reminders import reminder_manager
from app.system_control import execute_system_command
import re

dashboard_dir = Path(__file__).resolve().parent
templates_dir = dashboard_dir / "templates"
static_dir = dashboard_dir / "static"

app = FastAPI(title="FIXI Local Companion Dashboard", version=settings.APP_VERSION)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

frontend_dist = dashboard_dir.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dist), html=True), name="react_app")

templates = Jinja2Templates(directory=str(templates_dir))


class ChatMessageRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    speak_aloud: Optional[bool] = False


class SystemCommandRequest(BaseModel):
    text: str


class EpisodicMemoryRequest(BaseModel):
    user_id: str
    summary: str
    details: Optional[str] = ""
    category: Optional[str] = "general"
    importance: Optional[float] = 1.0


class FactRequest(BaseModel):
    user_id: str
    key: str
    value: str
    category: Optional[str] = "general"


class RobotActionRequest(BaseModel):
    action: str  # forward, backward, left, right, stop, wave
    speed: Optional[float] = 1.0
    duration: Optional[float] = 1.0
    angle: Optional[float] = 45.0


@app.on_event("startup")
async def startup_event():
    # Attach active running asyncio loop to EventBus
    event_bus.set_loop(asyncio.get_running_loop())
    # Start Fixi engine vision loop (disable background mic listener to prevent duplicate capture with browser)
    fixi_engine.start(start_voice=False)
    event_bus.emit(EventType.SYSTEM_STATUS_CHANGED, {"status": "online", "model": settings.OLLAMA_MODEL})


@app.on_event("shutdown")
def shutdown_event():
    fixi_engine.stop()


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """Real-time event stream for live dashboard and lab pages."""
    await event_bus.connect(websocket)
    try:
        while True:
            # Keep connection open; receive client heartbeat or commands
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        event_bus.disconnect(websocket)


@app.get("/api/events")
def get_event_history():
    """Returns recent events buffer for initialization."""
    return event_bus.get_history()


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    status = fixi_engine.get_system_status()
    users = user_manager.list_users()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "status": status,
            "users": users
        }
    )


@app.get("/api/status")
def get_status():
    return fixi_engine.get_system_status()


@app.get("/api/point-command/status")
def get_point_command_status():
    return fixi_engine.get_point_command_status()


@app.post("/api/system/command")
def system_command(payload: SystemCommandRequest):
    """Executes only allowlisted local commands for the recognized active user."""
    recognized = any(face.is_recognized for face in fixi_engine.latest_faces)
    if not fixi_engine.current_user_id or fixi_engine.recognition_confidence < 0.60 or not recognized:
        raise HTTPException(status_code=403, detail="A clearly recognized active user is required.")
    result = execute_system_command(payload.text, fixi_engine.current_user_id)
    if result.handled:
        fixi_engine.speak(result.message)
    return {
        "handled": result.handled,
        "message": result.message,
        "action": result.action,
        "reminder_id": result.reminder_id,
        "user_id": fixi_engine.current_user_id,
    }


@app.get("/api/reminders")
def list_active_user_reminders():
    if not fixi_engine.current_user_id:
        raise HTTPException(status_code=403, detail="No active recognized user.")
    return reminder_manager.list_for_user(fixi_engine.current_user_id)


def generate_mjpeg_frames():
    """Generates MJPEG video stream with bounding boxes and HUD overlays."""
    while True:
        success, frame = fixi_engine.camera.read_frame()
        if not success or frame is None:
            time.sleep(0.05)
            continue

        annotated = frame.copy()
        # Draw Person Detections (Green)
        for det in fixi_engine.latest_detections:
            x, y, w, h = det.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(
                annotated,
                f"Person: {int(det.confidence * 100)}%",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 0),
                2
            )

        # Draw Face Detections (Blue or Cyan)
        for face in fixi_engine.latest_faces:
            fx, fy, fw, fh = face.bbox
            color = (255, 180, 0) if face.is_recognized else (0, 140, 255)
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), color, 2)
            label = f"{face.matched_name} ({int(face.similarity_score * 100)}%)" if face.is_recognized else "Unknown Face"
            cv2.putText(
                annotated,
                label,
                (fx, max(20, fy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Point-and-command overlays remain image-space only.
        point_status = fixi_engine.get_point_command_status()
        for obj in point_status.get("objects", []):
            ox, oy, ow, oh = obj["bbox"]
            is_target = obj["object_id"] == point_status.get("target_id")
            color = (0, 255, 255) if is_target else (180, 180, 0)
            cv2.rectangle(annotated, (ox, oy), (ox + ow, oy + oh), color, 2)
            cv2.putText(annotated, f'{obj["label"]} {int(obj["confidence"] * 100)}%', (ox, max(20, oy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        gesture = point_status.get("gesture", {})
        fingertip = gesture.get("fingertip")
        direction = gesture.get("direction")
        if fingertip and direction and gesture.get("confidence", 0.0) >= 0.55:
            fh, fw = annotated.shape[:2]
            sx, sy = int(fingertip[0] * fw), int(fingertip[1] * fh)
            ex = int(sx + direction[0] * 180)
            ey = int(sy + direction[1] * 180)
            cv2.arrowedLine(annotated, (sx, sy), (ex, ey), (255, 255, 0), 2, tipLength=0.18)
            cv2.circle(annotated, (sx, sy), 6, (255, 255, 0), 2)

        cv2.putText(annotated, f'POINT: {point_status.get("state", "IDLE")} | {point_status.get("message", "")[:54]}', (12, annotated.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        # Encode JPEG
        ret, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
        time.sleep(0.04)  # ~25 FPS stream


@app.get("/api/camera/stream")
def camera_stream():
    return StreamingResponse(
        generate_mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/users")
def list_users():
    return user_manager.list_users()


@app.post("/api/users")
def create_user(
    name: str = Form(...),
    consent: str = Form("granted")
):
    status = ConsentStatus.GRANTED if consent == "granted" else ConsentStatus.DENIED
    user = user_manager.create_user(name=name, consent_status=status)
    memory_manager.set_user_fact(user["id"], "name", name)
    return user


@app.delete("/api/users/{user_id}")
def delete_user(user_id: str):
    success = user_manager.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    if fixi_engine.current_user_id == user_id:
        fixi_engine._reset_user_state()
    return {"deleted": True, "user_id": user_id}


@app.get("/api/memories/{user_id}")
def get_user_memories(user_id: str):
    """Returns isolated memories across all 3 levels for the specified user."""
    user = user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    facts = memory_manager.get_user_facts(user_id)
    episodic = memory_manager.list_episodic_memories(user_id)
    short_term = memory_manager.get_recent_conversation(user_id, limit=20)
    counts = memory_manager.get_memory_counts(user_id)

    return {
        "user": user,
        "facts": facts,
        "episodic": episodic,
        "short_term": short_term,
        "counts": counts
    }


@app.post("/api/memories/episodic")
def add_episodic_memory(payload: EpisodicMemoryRequest):
    mem_id = memory_manager.add_episodic_memory(
        user_id=payload.user_id,
        summary=payload.summary,
        details=payload.details or "",
        category=payload.category or "general",
        importance=payload.importance or 1.0
    )
    return {"id": mem_id, "user_id": payload.user_id, "summary": payload.summary}


@app.delete("/api/memories/episodic/{user_id}/{memory_id}")
def delete_episodic_memory(user_id: str, memory_id: int):
    success = memory_manager.delete_episodic_memory(user_id, memory_id)
    return {"deleted": success, "user_id": user_id, "memory_id": memory_id}


@app.post("/api/memories/facts")
def set_user_fact(payload: FactRequest):
    memory_manager.set_user_fact(payload.user_id, payload.key, payload.value, payload.category or "general")
    return {"saved": True, "key": payload.key, "value": payload.value}


@app.delete("/api/memories/facts/{user_id}/{key}")
def delete_user_fact(user_id: str, key: str):
    success = memory_manager.delete_user_fact(user_id, key)
    return {"deleted": success, "user_id": user_id, "key": key}


@app.delete("/api/memories/{user_id}/clear")
def clear_all_memories(user_id: str):
    memory_manager.clear_all_user_memories(user_id)
    return {"cleared": True, "user_id": user_id}


@app.post("/api/chat")
def chat_with_fixi(payload: ChatMessageRequest):
    """
    Dialogue endpoint for testing conversation directly from dashboard.
    If user_id is provided, overrides engine's active user context for testing.
    """
    if payload.user_id:
        user = user_manager.get_user(payload.user_id)
        if user:
            fixi_engine.current_user_id = user["id"]
            fixi_engine.current_user_name = user["name"]
            fixi_engine.state = FixiState.CONVERSING
            fixi_engine._last_person_seen_time = time.time()
            fixi_engine.last_interaction_time = time.time()
    elif not fixi_engine.current_user_id:
        users = user_manager.list_users()
        if len(users) == 1:
            fixi_engine.current_user_id = users[0]["id"]
            fixi_engine.current_user_name = users[0]["name"]
            fixi_engine.state = FixiState.CONVERSING
            fixi_engine._last_person_seen_time = time.time()
            fixi_engine.last_interaction_time = time.time()

    # Only speak through host sounddevice if explicitly requested (prevents double voice with browser audio)
    speak_aloud = bool(payload.speak_aloud)
    result = fixi_engine.handle_user_speech(payload.message, speak_aloud=speak_aloud)
    resp_text = result.get("response", "")
    result["audio_url"] = f"/api/tts?text={urllib.parse.quote(resp_text)}" if resp_text else ""
    return result


@app.post("/api/voice_chat")
async def voice_chat(
    request: Request,
    audio_file: Optional[UploadFile] = File(None),
    user_id: Optional[str] = Form(None),
    speak_aloud: Optional[bool] = Form(False)
):
    """
    Direct voice input endpoint. Accepts recorded audio (wav/webm/ogg/mp3),
    transcribes it locally with Faster-Whisper, feeds to Fixi AI with user memory,
    and returns transcript, AI response, and synthesized voice audio URL.
    """
    if audio_file is None:
        form = await request.form()
        candidate = form.get("audio_file") or form.get("file") or form.get("audio")
        if isinstance(candidate, UploadFile):
            audio_file = candidate
    if audio_file is None:
        raise HTTPException(status_code=400, detail="No recorded audio file was received.")

    if user_id:
        user = user_manager.get_user(user_id)
        if user:
            fixi_engine.current_user_id = user["id"]
            fixi_engine.current_user_name = user["name"]
            fixi_engine.state = FixiState.CONVERSING
            fixi_engine._last_person_seen_time = time.time()
            fixi_engine.last_interaction_time = time.time()
    elif not fixi_engine.current_user_id:
        users = user_manager.list_users()
        if len(users) == 1:
            fixi_engine.current_user_id = users[0]["id"]
            fixi_engine.current_user_name = users[0]["name"]
            fixi_engine.state = FixiState.CONVERSING
            fixi_engine._last_person_seen_time = time.time()
            fixi_engine.last_interaction_time = time.time()

    suffix = Path(audio_file.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        transcript = stt_engine.transcribe(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not transcript or not transcript.strip():
        fallback_msg = "I didn't catch that. Could you please speak again?"
        return {
            "transcript": "",
            "response": fallback_msg,
            "audio_url": f"/api/tts?text={urllib.parse.quote(fallback_msg)}"
        }

    # Route explicit OS-control phrases through the allowlisted local command layer.
    if re.search(r"\b(?:open|launch|start)\s+(?:the\s+)?(?:browser|web browser|terminal)\b|\bremind me to\b", transcript, re.IGNORECASE):
        recognized = any(face.is_recognized for face in fixi_engine.latest_faces)
        if not fixi_engine.current_user_id or fixi_engine.recognition_confidence < 0.60 or not recognized:
            response = "I need to clearly recognize you before running a system command."
            return {"transcript": transcript.strip(), "response": response, "audio_url": f"/api/tts?text={urllib.parse.quote(response)}"}
        control_result = execute_system_command(transcript, fixi_engine.current_user_id)
        if control_result.handled:
            fixi_engine.speak(control_result.message)
        return {
            "transcript": transcript.strip(),
            "response": control_result.message,
            "action": control_result.action,
            "audio_url": f"/api/tts?text={urllib.parse.quote(control_result.message)}",
        }

    # speak_aloud is False by default so only browser plays audio (no double voice)
    res = fixi_engine.handle_user_speech(transcript.strip(), speak_aloud=bool(speak_aloud))
    resp_text = res.get("response", "")
    return {
        "transcript": transcript.strip(),
        "response": resp_text,
        "extracted": res.get("extracted", {}),
        "user_id": fixi_engine.current_user_id,
        "audio_url": f"/api/tts?text={urllib.parse.quote(resp_text)}" if resp_text else ""
    }


@app.get("/api/tts")
def synthesize_tts(text: str):
    """Returns synthesized WAV audio for browser playback."""
    wav_bytes = tts_engine.synthesize_wav_bytes(text)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/api/robot/action")
def trigger_robot_action(payload: RobotActionRequest):
    """Controls simulated or hardware robot actuators."""
    act = payload.action.lower()
    if act == "forward":
        robot_controller.move_forward(payload.speed or 1.0, payload.duration or 1.0)
    elif act == "backward":
        robot_controller.move_backward(payload.speed or 1.0, payload.duration or 1.0)
    elif act == "left":
        robot_controller.turn_left(payload.angle or 45.0)
    elif act == "right":
        robot_controller.turn_right(payload.angle or 45.0)
    elif act == "stop":
        robot_controller.stop()
    elif act == "wave":
        robot_controller.wave()
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

    return robot_controller.get_state()


class GridGoalRequest(BaseModel):
    col: int
    row: int
    label: Optional[str] = "target"


class GridObstacleRequest(BaseModel):
    col: int
    row: int
    action: Optional[str] = "add"


@app.get("/api/robotics/state")
def get_robotics_state():
    """Returns 2D simulation state, grid occupancy, robot odometry, and FSM status."""
    return robotics_sim.get_full_state()


@app.post("/api/robotics/goal")
def set_robotics_goal(payload: GridGoalRequest):
    """Sets a target cell and triggers 8-connected A* path planning."""
    success = robotics_sim.set_goal(payload.col, payload.row, label=payload.label or "target")
    return {"success": success, "state": robotics_sim.get_full_state()}


@app.post("/api/robotics/navigate_person")
def navigate_to_person():
    """Navigates to safe interaction zone adjacent to detected person."""
    if robotics_sim.person_col is not None and robotics_sim.person_row is not None:
        c, r = robotics_sim.person_col, robotics_sim.person_row
    elif fixi_engine.latest_detections:
        c, r = fixi_engine.latest_detections[0].grid_x, fixi_engine.latest_detections[0].grid_y
    else:
        c, r = 10, 14
    success = robotics_sim.navigate_to_person(c, r)
    return {"success": success, "target": [c, r], "state": robotics_sim.get_full_state()}


@app.post("/api/robotics/follow_person")
def follow_person():
    """Starts simulation-only follow mode using the latest camera person position."""
    recognized = any(face.is_recognized for face in fixi_engine.latest_faces)
    if not fixi_engine.current_user_id or fixi_engine.recognition_confidence < 0.60 or not recognized:
        return {
            "success": False,
            "message": "A clearly recognized active user is required before following.",
            "state": robotics_sim.get_full_state(),
        }
    if not fixi_engine.latest_detections:
        return {
            "success": False,
            "message": "No person is currently detected.",
            "state": robotics_sim.get_full_state(),
        }
    detection = fixi_engine.latest_detections[0]
    success = robotics_sim.start_following(detection.grid_x, detection.grid_y)
    return {
        "success": success,
        "target": [detection.grid_x, detection.grid_y],
        "state": robotics_sim.get_full_state(),
    }


@app.post("/api/robotics/step")
def step_robotics_simulation():
    """Advances one step along path with closed-loop verification."""
    res = robotics_sim.step()
    return {"result": res, "state": robotics_sim.get_full_state()}


@app.post("/api/robotics/obstacle")
def toggle_robotics_obstacle(payload: GridObstacleRequest):
    """Adds or removes dynamic obstacle on the occupancy grid."""
    if payload.action == "remove":
        robotics_sim.remove_obstacle(payload.col, payload.row)
    else:
        robotics_sim.add_obstacle(payload.col, payload.row)
    return {"success": True, "state": robotics_sim.get_full_state()}


@app.post("/api/robotics/emergency_stop")
def robotics_emergency_stop():
    """Safety stop halting all autonomous navigation."""
    robotics_sim.emergency_stop()
    return {"stopped": True, "state": robotics_sim.get_full_state()}


@app.post("/api/robotics/reset")
def robotics_reset():
    """Resets 2D simulation to home coordinates and clears dynamic obstacles."""
    robotics_sim.reset()
    return {"reset": True, "state": robotics_sim.get_full_state()}


@app.get("/api/evaluation/benchmarks")
def get_evaluation_benchmarks():
    """Academic benchmark suite evaluating CV and Intelligent Robotics."""
    cv_metrics = fixi_engine.person_detector.get_metrics()
    robotics_benchmarks = run_robotics_benchmarks()
    return {
        "computer_vision": cv_metrics,
        "intelligent_robotics": robotics_benchmarks
    }


class RegisterFaceRequest(BaseModel):
    name: str


class RegisterFaceImageRequest(BaseModel):
    name: str
    image_base64: str


@app.post("/api/register_face")
def register_face(payload: RegisterFaceRequest):
    """Captures face in live camera view and registers user profile with voice output."""
    # speak_aloud=False so audio confirmation is played cleanly by the browser without double output
    result = fixi_engine.register_face_manually(payload.name, speak_aloud=False)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    msg = result.get("message", "")
    result["audio_url"] = f"/api/tts?text={urllib.parse.quote(msg)}" if msg else ""
    return result


@app.post("/api/register_face_image")
def register_face_image(payload: RegisterFaceImageRequest):
    """Registers face from a base64 encoded image (e.g. browser webcam snapshot)."""
    raw_b64 = payload.image_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(raw_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    result = fixi_engine.register_face_from_image(payload.name, img_bytes, speak_aloud=False)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    msg = result.get("message", "")
    result["audio_url"] = f"/api/tts?text={urllib.parse.quote(msg)}" if msg else ""
    return result


@app.post("/api/register_face_upload")
async def register_face_upload(
    name: str = Form(...),
    photo: UploadFile = File(...)
):
    """Registers face from an uploaded photo file."""
    img_bytes = await photo.read()
    result = fixi_engine.register_face_from_image(name, img_bytes, speak_aloud=False)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    msg = result.get("message", "")
    result["audio_url"] = f"/api/tts?text={urllib.parse.quote(msg)}" if msg else ""
    return result
