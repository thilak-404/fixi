"""
FIXI - Local AI Person-Aware Robotic Companion
Main Intelligence Engine coordinating Vision, Speech, Memory, LLM, and Robotics.
Runs 100% locally with zero cloud dependencies.
"""

import queue
import time
import threading
from typing import Optional, Dict, Any, List, Callable, Union
import cv2
import numpy as np

from app.config.settings import settings
from app.vision.camera import BaseCamera, WebcamCamera, SyntheticCamera, AdbMotorolaCamera
from app.vision.person_detector import BasePersonDetector, HOGPersonDetector, SimulatedPersonDetector, PersonDetection
from app.vision.face_recognition import (
    BaseFaceRecognizer,
    OpenCVYuNetSFaceRecognizer,
    SimulatedFaceRecognizer,
    FaceInfo,
    save_face_crop
)
from app.speech.stt import BaseSTT, stt_engine, MockSTT
from app.speech.tts import BaseTTS, tts_engine, MockTTS
from app.speech.vad import EnergyVAD
from app.ai.llm import BaseLLMProvider, llm_provider
from app.ai.response import ResponseGenerator, response_generator
from app.users.user_manager import user_manager
from app.users.consent import (
    ConsentStatus,
    CONSENT_REQUEST_PHRASE,
    NAME_REQUEST_PHRASE,
    DENIED_ACKNOWLEDGEMENT_PHRASE,
    parse_consent_response
)
from app.memory.memory_manager import memory_manager
from app.robotics.controller import robot_controller, BaseRobotController
from app.robotics.pointing import MediaPipeHandTracker, PointingGestureRecognizer, GestureClass, GestureObservation
from app.robotics.object_detection import LocalYOLOObjectDetector
from app.robotics.target_selection import ObjectDetection
from app.robotics.point_command_engine import PointCommandEngine, PointCommandStatus, PointCommandState
from app.robotics.commands import CommandIntent, PointCommandParser
from app.events import event_bus
from app.robotics.simulation import robotics_sim
from app.memory.reminders import reminder_manager


class FixiState:
    IDLE = "IDLE"
    PERSON_DETECTED = "PERSON_DETECTED"
    AWAITING_CONSENT = "AWAITING_CONSENT"
    AWAITING_NAME = "AWAITING_NAME"
    CONVERSING = "CONVERSING"
    SPEAKING = "SPEAKING"


class FixiEngine:
    """
    Central Coordinator for Fixi.
    Executes the continuous Person-Aware + Memory-Aware intelligence loop.
    """
    def __init__(
        self,
        camera: Optional[BaseCamera] = None,
        person_detector: Optional[BasePersonDetector] = None,
        face_recognizer: Optional[BaseFaceRecognizer] = None,
        stt: Optional[BaseSTT] = None,
        tts: Optional[BaseTTS] = None,
        llm: Optional[BaseLLMProvider] = None,
        robot: Optional[BaseRobotController] = None,
        user_mgr: Optional[Any] = None,
        mem_mgr: Optional[Any] = None
    ):
        if camera is not None:
            self.camera = camera
        elif AdbMotorolaCamera.is_device_connected():
            self.camera = AdbMotorolaCamera()
        else:
            self.camera = WebcamCamera()
        self.person_detector = person_detector or HOGPersonDetector()
        self.face_recognizer = face_recognizer or OpenCVYuNetSFaceRecognizer()
        self.stt = stt or stt_engine
        self.tts = tts or tts_engine
        self.llm = llm or llm_provider
        self.robot = robot or robot_controller
        self.user_manager = user_mgr or user_manager
        self.memory_manager = mem_mgr or memory_manager

        self.responder = ResponseGenerator(llm=self.llm)
        self.responder.memory_manager = self.memory_manager
        self.responder.extractor.memory_manager = self.memory_manager
        self.vad = EnergyVAD()
        self.hand_tracker = MediaPipeHandTracker()
        self.gesture_recognizer = PointingGestureRecognizer()
        self.object_detector = LocalYOLOObjectDetector()
        self.point_command_engine = PointCommandEngine(robotics_sim)
        self.command_parser = PointCommandParser()

        # State Tracking
        self.state = FixiState.IDLE
        self.current_user_id: Optional[str] = None
        self.current_user_name: Optional[str] = None
        self.current_user_consent: Optional[ConsentStatus] = None
        self.current_face_crop: Optional[np.ndarray] = None
        self.current_face_embedding: Optional[np.ndarray] = None

        self.latest_detections: List[PersonDetection] = []
        self.latest_faces: List[FaceInfo] = []
        self.last_interaction_time: float = time.time()
        self.recognition_confidence: float = 0.0
        self.latest_gesture: GestureObservation = self.gesture_recognizer.last_observation
        self.latest_objects: List[ObjectDetection] = []
        self.latest_point_command: PointCommandStatus = self.point_command_engine.status

        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.Lock()

        # Cooldowns to prevent repetitive speech loops
        self._last_greeting_time: Dict[str, float] = {}
        self._last_unknown_prompt_time: float = 0.0
        self._last_person_seen_time: float = time.time()

        # Continuous background voice listening
        self.voice_listening_active: bool = False
        self._voice_thread: Optional[threading.Thread] = None

    def add_event_listener(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self._callbacks.append(cb)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        payload = {
            "event": event_type,
            "state": self.state,
            "user_id": self.current_user_id,
            "user_name": self.current_user_name,
            "timestamp": time.time(),
            **data
        }
        for cb in self._callbacks:
            try:
                cb(payload)
            except Exception:
                pass

    def start(self, start_voice: bool = True) -> bool:
        """Starts the Fixi camera, vision loop, and continuous voice listener."""
        if self._running:
            return True
        cam_ok = self.camera.start()
        if not cam_ok and AdbMotorolaCamera.is_device_connected() and not isinstance(self.camera, AdbMotorolaCamera):
            self.camera = AdbMotorolaCamera()
            cam_ok = self.camera.start()

        if not cam_ok:
            print("Physical camera unavailable; activating SyntheticCamera.")
            try:
                self.camera.stop()
            except Exception:
                pass
            self.camera = SyntheticCamera()
            self.camera.start()

        self._running = True
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        # Start continuous hands-free voice listener if requested
        if start_voice:
            self.start_voice_listener()
        return True

    def stop(self) -> None:
        self._running = False
        self.stop_voice_listener()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=1.0)
        self.camera.stop()

    def _run_loop(self) -> None:
        """Continuous background vision processing loop."""
        while self._running:
            if isinstance(self.camera, SyntheticCamera):
                now = time.time()
                if now - getattr(self, "_last_cam_check_time", 0.0) > 4.0:
                    self._last_cam_check_time = now
                    try:
                        probe_cam = AdbMotorolaCamera() if AdbMotorolaCamera.is_device_connected() else WebcamCamera()
                        if probe_cam.start():
                            print("Live physical camera detected! Upgrading from SyntheticCamera to live camera.")
                            self.camera.stop()
                            self.camera = probe_cam
                    except Exception:
                        pass

            success, frame = self.camera.read_frame()
            if success and frame is not None:
                self.process_vision_frame(frame, check_timeout=True)
            time.sleep(settings.FRAME_PROCESS_INTERVAL)

    def process_vision_frame(self, frame: np.ndarray, check_timeout: bool = False) -> Dict[str, Any]:
        """
        Processes a single vision frame:
        Detects persons -> Recognizes faces -> Updates robot gaze -> Handles identity.
        """
        with self._lock:
            # 1. Person Detection
            detections = self.person_detector.detect(frame)
            self.latest_detections = detections

            # 2. Face Recognition
            faces = self.face_recognizer.detect_and_recognize(frame)
            self.latest_faces = faces

            # Point-and-command perception stays image-space only. Models are optional;
            # unavailable hand/object detectors produce empty observations safely.
            hands = self.hand_tracker.detect(frame)
            hand = max(hands, key=lambda item: item.confidence) if hands else None
            self.latest_gesture = self.gesture_recognizer.update(hand)
            self.latest_objects = self.object_detector.detect(frame)
            previous_point_state = self.latest_point_command.state
            self.latest_point_command = self.point_command_engine.observe(
                self.latest_gesture, self.latest_objects
            )
            if self.latest_point_command.state != previous_point_state:
                event_bus.emit(
                    "point_command_state_changed",
                    self.get_point_command_status(),
                )

            # If no people or faces detected
            if not detections and not faces:
                now = time.time()
                is_recent_interaction = (now - self.last_interaction_time < 60.0)
                should_reset = not check_timeout or ((now - self._last_person_seen_time > 45.0) and not is_recent_interaction)
                if should_reset:
                    if self.state not in [FixiState.IDLE, FixiState.SPEAKING]:
                        # User walked away
                        self._emit_event("person_left", {"previous_user": self.current_user_name})
                        self._reset_user_state()
                        self.state = FixiState.IDLE
                return {"people_count": 0, "state": self.state}

            self._last_person_seen_time = time.time()

            # People detected
            primary_person = detections[0] if detections else None
            if primary_person:
                # Direct robot head / gaze toward person
                self.robot.look_at_person(primary_person.bbox, frame_size=(frame.shape[1], frame.shape[0]))
                # Ground-plane projection to 2D occupancy grid
                try:
                    robotics_sim.update_person_position(primary_person.grid_x, primary_person.grid_y)
                except Exception:
                    pass

            # 3. Check for recognized or unknown face
            if faces:
                face = faces[0]
                self.recognition_confidence = face.similarity_score

                # Crop face region
                fx, fy, fw, fh = face.bbox
                h, w = frame.shape[:2]
                fx, fy = max(0, fx), max(0, fy)
                fw, fh = min(w - fx, fw), min(h - fy, fh)
                if fw > 20 and fh > 20:
                    self.current_face_crop = frame[fy:fy+fh, fx:fx+fw].copy()
                if face.embedding is not None:
                    self.current_face_embedding = face.embedding

                if face.is_recognized and face.matched_user_id:
                    # Recognized known user
                    now = time.time()
                    is_new_user = self.current_user_id != face.matched_user_id
                    time_since_last_greet = now - self._last_greeting_time.get(face.matched_user_id, 0.0)

                    if is_new_user:
                        self.current_user_id = face.matched_user_id
                        self.current_user_name = face.matched_name
                        self.current_user_consent = ConsentStatus.GRANTED
                        self.state = FixiState.CONVERSING
                        self.user_manager.update_last_seen(face.matched_user_id)
                        self.last_interaction_time = now

                    # Verbally greet recognized user (if new encounter or > 45s since last greeting)
                    if is_new_user or time_since_last_greet > 45.0:
                        greeting = f"Hi {face.matched_name}! Good to see you again."
                        self._last_greeting_time[face.matched_user_id] = now
                        self._emit_event("user_recognized", {
                            "user_id": face.matched_user_id,
                            "user_name": face.matched_name,
                            "confidence": face.similarity_score,
                            "greeting": greeting
                        })
                        self.speak(greeting)

                    self._deliver_due_reminders(face.matched_user_id, face.matched_name or "there")
                else:
                    # Unknown person
                    now = time.time()
                    if self.current_user_id is None and self.state == FixiState.IDLE:
                        if now - self._last_unknown_prompt_time > 30.0:
                            self.state = FixiState.AWAITING_CONSENT
                            self.current_user_consent = ConsentStatus.PENDING
                            self._last_unknown_prompt_time = now
                            self._emit_event("unknown_person_detected", {
                                "prompt": CONSENT_REQUEST_PHRASE
                            })
                            self.speak(CONSENT_REQUEST_PHRASE)
            else:
                # Person detected but face not clearly visible
                if self.state == FixiState.IDLE:
                    self.state = FixiState.PERSON_DETECTED

            return {
                "people_count": len(detections),
                "faces_count": len(faces),
                "current_user": self.current_user_name,
                "state": self.state
            }

    def handle_user_speech(self, user_text: str, speak_aloud: bool = True) -> Dict[str, Any]:
        """
        Main dialogue handler when user speaks (via microphone or web dashboard).
        Handles consent flow, onboarding, and personalized memory conversation.
        """
        text = user_text.strip()
        if not text:
            return {"response": "", "state": self.state}

        self.last_interaction_time = time.time()

        # STATE 1: Awaiting Consent from Unknown Person
        if self.state == FixiState.AWAITING_CONSENT:
            answered, is_granted = parse_consent_response(text)
            if answered:
                if is_granted:
                    self.current_user_consent = ConsentStatus.GRANTED
                    self.state = FixiState.AWAITING_NAME
                    resp = NAME_REQUEST_PHRASE
                    if speak_aloud:
                        self.speak(resp)
                    return {"response": resp, "state": self.state}
                else:
                    self.current_user_consent = ConsentStatus.DENIED
                    self.state = FixiState.CONVERSING
                    self.current_user_id = None
                    self.current_user_name = "Guest"
                    resp = DENIED_ACKNOWLEDGEMENT_PHRASE
                    if speak_aloud:
                        self.speak(resp)
                    return {"response": resp, "state": self.state}
            else:
                resp = "I didn't catch that. Would you like me to remember you for future conversations?"
                if speak_aloud:
                    self.speak(resp)
                return {"response": resp, "state": self.state}

        # STATE 2: Awaiting Name from Consenting New User
        elif self.state == FixiState.AWAITING_NAME:
            # Extract name using rule heuristics
            import re
            name_match = re.search(r"\bmy name is ([A-Za-z]+)\b", text, re.IGNORECASE) or \
                         re.search(r"\bi am ([A-Za-z]+)\b", text, re.IGNORECASE) or \
                         re.search(r"\bi'm ([A-Za-z]+)\b", text, re.IGNORECASE) or \
                         re.search(r"^([A-Za-z]+)$", text)
            if name_match:
                extracted_name = name_match.group(1).capitalize()
            else:
                words = text.split()
                extracted_name = words[-1].strip(".,!").capitalize() if words else "Friend"

            # Create User Profile in SQLite
            new_user = self.user_manager.create_user(
                name=extracted_name,
                face_embedding=self.current_face_embedding,
                consent_status=ConsentStatus.GRANTED
            )
            self.current_user_id = new_user["id"]
            self.current_user_name = new_user["name"]
            self.state = FixiState.CONVERSING

            # Save face thumbnail image if crop is available
            if self.current_face_crop is not None:
                save_face_crop(self.current_face_crop, self.current_user_id)

            # Store name as first profile fact
            self.memory_manager.set_user_fact(self.current_user_id, "name", extracted_name)

            # Robot wave gesture to welcome new user!
            self.robot.wave()

            resp = f"Nice to meet you, {extracted_name}!"
            if speak_aloud:
                self.speak(resp)
            return {"response": resp, "user_id": self.current_user_id, "state": self.state}

        # STATE 3: Active Personalized Conversation or Robotics Commands
        else:
            self.state = FixiState.CONVERSING
            parsed_command = self.command_parser.parse(text, user_id=self.current_user_id)
            if parsed_command.intent in {
                CommandIntent.GO,
                CommandIntent.RETURN_HOME,
                CommandIntent.STOP,
                CommandIntent.CANCEL,
            }:
                point_status = self.point_command_engine.accept_command(
                    parsed_command,
                    active_user_id=self.current_user_id,
                    recognition_confidence=self.recognition_confidence,
                    people_count=len(self.latest_detections),
                    face_is_recognized=any(face.is_recognized for face in self.latest_faces),
                )
                self.latest_point_command = point_status
                event_bus.emit("command_validated", self.get_point_command_status())
                if parsed_command.intent in {CommandIntent.STOP, CommandIntent.CANCEL} or point_status.state != PointCommandState.ERROR:
                    response = point_status.message
                    if speak_aloud:
                        self.speak(response)
                    return {
                        "response": response,
                        "user_id": self.current_user_id,
                        "action_executed": parsed_command.intent.value,
                        "is_robotics": True,
                        "point_command": self.get_point_command_status(),
                        "state": self.state,
                    }
            from app.ai.intent import intent_classifier
            coords = None
            if self.latest_detections:
                coords = (self.latest_detections[0].grid_x, self.latest_detections[0].grid_y)

            gen_res = intent_classifier.process_utterance(
                text=text,
                user_id=self.current_user_id,
                user_name=self.current_user_name,
                person_spatial_coords=coords,
                responder=self.responder,
                recognition_confidence=self.recognition_confidence,
                people_count=len(self.latest_detections),
                face_is_recognized=any(face.is_recognized for face in self.latest_faces),
            )
            resp = gen_res.get("response", "")
            if speak_aloud:
                self.speak(resp)
            return {
                "response": resp,
                "user_id": self.current_user_id,
                "extracted": gen_res.get("extracted", {}),
                "action_executed": gen_res.get("action_executed"),
                "is_robotics": gen_res.get("is_robotics", False),
                "state": self.state
            }

    def get_point_command_status(self) -> Dict[str, Any]:
        """Returns a JSON-safe snapshot for the HUD and API."""
        status = self.latest_point_command
        selection = status.selection
        candidates = []
        if selection:
            candidates = [
                {
                    "object_id": candidate.detection.object_id,
                    "label": candidate.detection.label,
                    "confidence": round(candidate.detection.confidence, 3),
                    "bbox": list(candidate.detection.bbox),
                    "score": round(candidate.score, 3),
                }
                for candidate in selection.candidates
            ]
        return {
            "state": status.state.value,
            "message": status.message,
            "target_id": status.target_id,
            "target_label": status.target_label,
            "target_grid": list(status.target_grid) if status.target_grid else None,
            "selection_status": selection.status if selection else "none",
            "candidates": candidates,
            "gesture": {
                "class": self.latest_gesture.gesture.value,
                "confidence": round(self.latest_gesture.confidence, 3),
                "stable_frames": self.latest_gesture.stable_frames,
                "fingertip": list(self.latest_gesture.fingertip) if self.latest_gesture.fingertip else None,
                "direction": list(self.latest_gesture.direction) if self.latest_gesture.direction else None,
            },
            "objects": [
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "confidence": round(obj.confidence, 3),
                    "bbox": list(obj.bbox),
                }
                for obj in self.latest_objects
            ],
        }

    def speak(self, text: str, blocking: bool = False) -> None:
        """Speaks response using local TTS through computer speaker without freezing vision loop."""
        if not text or not text.strip():
            return

        def _do_speak():
            prev_state = self.state
            self.state = FixiState.SPEAKING
            try:
                self.tts.speak(text)
            except Exception as e:
                print(f"TTS audio playback error: {e}")
            finally:
                self.state = prev_state

        if blocking:
            _do_speak()
        else:
            threading.Thread(target=_do_speak, daemon=True).start()

    def _deliver_due_reminders(self, user_id: str, user_name: str) -> None:
        for reminder in reminder_manager.claim_due(user_id):
            message = f"Reminder for {user_name}: {reminder['message']}"
            event_bus.emit("reminder_due", {
                "reminder_id": reminder["id"],
                "user_id": user_id,
                "message": reminder["message"],
            })
            self.speak(message)

    def register_face_manually(self, name: str, speak_aloud: bool = True) -> Dict[str, Any]:
        """
        Registers the face currently in front of the camera as a new user.
        Actively acquires fresh frames from camera if needed.
        Generates 128D embedding, saves face crop, and announces confirmation aloud.
        """
        with self._lock:
            found_embedding = None
            found_crop = None

            # 1. First check if we already have a clean face embedding cached
            if self.current_face_embedding is not None:
                found_embedding = self.current_face_embedding
                found_crop = self.current_face_crop

            # 2. If not, actively scan camera frames over up to 1.5 seconds
            if found_embedding is None:
                for _ in range(12):
                    ok, fr = self.camera.read_frame()
                    if ok and fr is not None and fr.size > 0:
                        faces = self.face_recognizer.detect_and_recognize(fr)
                        if faces and faces[0].embedding is not None:
                            found_embedding = faces[0].embedding
                            fx, fy, fw, fh = faces[0].bbox
                            h, w = fr.shape[:2]
                            fx, fy = max(0, fx), max(0, fy)
                            fw, fh = min(w - fx, fw), min(h - fy, fh)
                            if fw > 20 and fh > 20:
                                found_crop = fr[fy:fy+fh, fx:fx+fw].copy()
                            break
                    time.sleep(0.12)

            if found_embedding is None:
                msg = "No face clearly detected. Please ensure your face is well-lit and centered in front of the camera."
                if speak_aloud:
                    self.speak(msg)
                return {"success": False, "message": msg}

            user_name = name.strip().capitalize()
            new_user = self.user_manager.create_user(
                name=user_name,
                face_embedding=found_embedding,
                consent_status=ConsentStatus.GRANTED
            )
            uid = new_user["id"]

            if found_crop is not None:
                save_face_crop(found_crop, uid)
                self.current_face_crop = found_crop

            self.current_face_embedding = found_embedding
            self.memory_manager.set_user_fact(uid, "name", user_name)
            self.current_user_id = uid
            self.current_user_name = user_name
            self.current_user_consent = ConsentStatus.GRANTED
            self.state = FixiState.CONVERSING
            self.recognition_confidence = 0.99
            self._last_greeting_time[uid] = time.time()
            self._last_person_seen_time = time.time()

            # Robot celebration wave
            self.robot.wave()

            confirm_msg = f"Nice to meet you, {user_name}! I have registered your face and created your profile."
            if speak_aloud:
                self.speak(confirm_msg)
            return {
                "success": True,
                "user_id": uid,
                "name": user_name,
                "message": confirm_msg
            }

    def register_face_from_image(self, name: str, image_bytes_or_bgr: Union[bytes, np.ndarray], speak_aloud: bool = True) -> Dict[str, Any]:
        """
        Registers a face directly from image bytes or BGR array (e.g. browser webcam snapshot or uploaded photo).
        """
        if isinstance(image_bytes_or_bgr, bytes):
            nparr = np.frombuffer(image_bytes_or_bgr, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            frame = image_bytes_or_bgr

        if frame is None or frame.size == 0:
            return {"success": False, "message": "Invalid image data provided."}

        faces = self.face_recognizer.detect_and_recognize(frame)
        if not faces or faces[0].embedding is None:
            msg = "No face found in the provided image. Please ensure your face is clearly visible."
            if speak_aloud:
                self.speak(msg)
            return {"success": False, "message": msg}

        face = faces[0]
        found_embedding = face.embedding
        fx, fy, fw, fh = face.bbox
        h, w = frame.shape[:2]
        fx, fy = max(0, fx), max(0, fy)
        fw, fh = min(w - fx, fw), min(h - fy, fh)
        found_crop = frame[fy:fy+fh, fx:fx+fw].copy() if fw > 20 and fh > 20 else frame.copy()

        with self._lock:
            user_name = name.strip().capitalize()
            new_user = self.user_manager.create_user(
                name=user_name,
                face_embedding=found_embedding,
                consent_status=ConsentStatus.GRANTED
            )
            uid = new_user["id"]
            save_face_crop(found_crop, uid)

            self.current_face_crop = found_crop
            self.current_face_embedding = found_embedding
            self.memory_manager.set_user_fact(uid, "name", user_name)
            self.current_user_id = uid
            self.current_user_name = user_name
            self.current_user_consent = ConsentStatus.GRANTED
            self.state = FixiState.CONVERSING
            self.recognition_confidence = 0.99
            self._last_greeting_time[uid] = time.time()
            self._last_person_seen_time = time.time()

            # Push this frame into camera so HUD shows the registered face
            if hasattr(self.camera, "push_external_frame"):
                self.camera.push_external_frame(frame)

            self.robot.wave()
            confirm_msg = f"Nice to meet you, {user_name}! I have registered your face from the photo."
            if speak_aloud:
                self.speak(confirm_msg)
            return {
                "success": True,
                "user_id": uid,
                "name": user_name,
                "message": confirm_msg
            }

    def start_voice_listener(self) -> bool:
        """Starts continuous microphone listening using EnergyVAD and Faster-Whisper."""
        if self.voice_listening_active:
            return True
        self.voice_listening_active = True
        self._voice_thread = threading.Thread(target=self._voice_listen_loop, daemon=True)
        self._voice_thread.start()
        return True

    def stop_voice_listener(self) -> None:
        self.voice_listening_active = False

    def _voice_listen_loop(self) -> None:
        """Continuously listens to default mic, detects speech with VAD, and transcribes."""
        try:
            import sounddevice as sd
            chunk_size = 1600  # 100ms chunks at 16kHz
            audio_buffer: List[np.ndarray] = []

            with sd.InputStream(samplerate=settings.AUDIO_SAMPLE_RATE, channels=1, dtype="int16") as stream:
                while self.voice_listening_active and self._running:
                    # Do not listen to our own voice output (anti-echo)
                    if self.state == FixiState.SPEAKING:
                        time.sleep(0.1)
                        continue

                    data, _ = stream.read(chunk_size)
                    chunk = data.flatten()
                    speaking, ended = self.vad.process_chunk(chunk)

                    if speaking:
                        audio_buffer.append(chunk)
                    elif ended and audio_buffer:
                        full_audio = np.concatenate(audio_buffer)
                        audio_buffer = []
                        if len(full_audio) >= int(settings.AUDIO_SAMPLE_RATE * 0.5):
                            text = self.stt.transcribe(full_audio)
                            if text and text.strip():
                                print(f"🎙️ Mic Heard: '{text.strip()}'")
                                self.handle_user_speech(text.strip())
                    else:
                        if len(audio_buffer) > 100:
                            audio_buffer = []
        except Exception as e:
            print(f"Notice: Background microphone listener stopped: {e}")
            self.voice_listening_active = False

    def _reset_user_state(self) -> None:
        self.current_user_id = None
        self.current_user_name = None
        self.current_user_consent = None
        self.current_face_crop = None
        self.current_face_embedding = None
        self.recognition_confidence = 0.0

    def get_system_status(self) -> Dict[str, Any]:
        """Returns comprehensive diagnostic and status info for dashboard."""
        return {
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "state": self.state,
            "current_person": self.current_user_name or "None",
            "current_user_id": self.current_user_id or "None",
            "recognition_confidence": round(self.recognition_confidence * 100, 1),
            "camera_active": self.camera.is_active(),
            "camera_source": "synthetic" if isinstance(self.camera, SyntheticCamera) else ("motorola_adb" if isinstance(self.camera, AdbMotorolaCamera) else "webcam"),
            "microphone_active": True,
            "llm_provider": "Ollama (Local)",
            "llm_model": settings.OLLAMA_MODEL,
            "llm_status": "Active" if self.llm.is_available() else "Offline",
            "tts_provider": "Piper TTS (Local)",
            "stt_provider": "Faster-Whisper (Local)",
            "memory_storage": "Local SQLite (WAL)",
            "internet_dependency": "None (100% Local)",
            "robot_state": self.robot.get_state(),
            "last_interaction": int(time.time() - self.last_interaction_time)
        }


fixi_engine = FixiEngine()
