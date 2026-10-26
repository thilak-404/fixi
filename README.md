# FIXI — Local AI Person-Aware Robotic Companion

Fixi is a fully local, privacy-focused intelligent robotic companion. It detects individuals through computer vision, identifies who they are, loads their isolated memories, communicates naturally using voice, learns new preferences and facts from the conversation, stores them locally, and uses them in future interactions.

**Fixi does NOT depend on cloud AI APIs.**
- 🚫 No OpenAI API
- 🚫 No Gemini API
- 🚫 No cloud speech / cloud TTS API
- 🚫 No cloud database
- 🚫 No remote LLM API
- 🔒 **100% on-device AI inference, speech processing, and SQLite memory.**

---

## 1. Core Architecture

The central intelligence loop of Fixi operates continuously:

```text
Camera (Webcam / /dev/video*)
   ↓
Person Detection (HOG / NMS / Bounding Boxes)
   ↓
Person Recognition (YuNet + SFace 128D Cosine Embeddings)
   ↓
Identify User (Sequential IDs: user_001, user_002, etc.)
   ↓
Load User-Specific Memory (Level A: Short-Term, Level B: Episodic, Level C: Facts)
   ↓
Listen to User (Microphone + Energy VAD)
   ↓
Speech-to-Text (Faster-Whisper local CTranslate2)
   ↓
Understand Intent & Context
   ↓
Retrieve Relevant Memory (Cosine vector similarity & keyword boosting)
   ↓
Local LLM (Ollama: Qwen2.5 / Llama3.2 on local GPU/CPU)
   ↓
Generate Personalized Response (Calm, friendly, concise, conversational)
   ↓
Text-to-Speech (Piper Neural TTS - 100% offline)
   ↓
Speaker / Audio Stream
   ↓
Extract Important New Memories (Facts, learning goals, preferences)
   ↓
Store Memory Locally (SQLite with strict user_id isolation)
```

---

## 2. Three Levels of Memory & Strict Isolation

Memory is structured into 3 distinct tiers:

1. **Level A: Short-Term Memory**
   - Retains the immediate multi-turn conversation context for the active session.
   - Enables coherent follow-up questions and conversational continuity.

2. **Level B: Episodic Memory**
   - Semantically indexed records of meaningful past interactions, events, and explicit requests.
   - Vector-embedded locally (128-dimensional L2-normalized embeddings) and queried via cosine similarity with keyword boosting.

3. **Level C: User Profile Memory (Facts)**
   - Stable key-value facts about the user (e.g. `name`, `interest`, `learning`, `occupation`, `preferences`).
   - Dynamically injected into Fixi's system prompt whenever the user is recognized.

### Strict Memory Isolation Guarantee
User A must never receive User B's memories.
- Every database query strictly filters by `user_id`.
- SQLite enforces foreign keys with `ON DELETE CASCADE`.
- Vector similarity searches are restricted exclusively to the active user's partition.
- Deleting a user permanently purges their profile, face embeddings, thumbnail images, conversation turns, episodic memories, and facts.

---

## 3. Technology Stack

| Component | Technology | Offline / Local Status |
| :--- | :--- | :--- |
| **Vision Detector** | OpenCV HOG + Non-Maximum Suppression | 100% Local / Built-in |
| **Face Recognition** | OpenCV YuNet + SFace (ONNX Zoo) | 100% Local (128D embeddings) |
| **Speech-to-Text (STT)**| Faster-Whisper (tiny.en / base.en) | 100% Local (CTranslate2) |
| **Text-to-Speech (TTS)**| Piper Neural TTS (en_US-lessac-medium) | 100% Local ONNX |
| **Voice Activity (VAD)**| Adaptive RMS Energy Detector | 100% Local |
| **Local LLM** | Ollama (`qwen2.5:0.5b` / `llama3.2`) | 100% Local (GPU/CPU) |
| **Vector Retrieval** | Local fast subword hashing + Cosine similarity | 100% Local |
| **Database** | SQLite3 with WAL mode | 100% Local File |
| **Dashboard** | FastAPI + HTML5 / CSS3 / JavaScript | 100% Local Web Server |
| **Robotics** | Hardware Abstraction Layer (`RobotController`) | Simulation & Physical Ready |

---

## 4. Project Structure

```text
FIXI/
├── app/
│   ├── main.py                  # FixiEngine coordinator (main loop)
│   ├── vision/
│   │   ├── camera.py            # Webcam & Synthetic camera capture
│   │   ├── person_detector.py   # HOG & cascade person detectors
│   │   └── face_recognition.py  # YuNet detection & SFace embeddings
│   ├── speech/
│   │   ├── stt.py               # Local Faster-Whisper STT
│   │   ├── tts.py               # Local Piper Neural TTS
│   │   └── vad.py               # Voice Activity Detection
│   ├── ai/
│   │   ├── llm.py               # OllamaProvider & LLM abstraction
│   │   ├── prompts.py           # Persona and prompt templates
│   │   └── response.py          # Response generation & extraction trigger
│   ├── memory/
│   │   ├── database.py          # SQLite database connection & schema
│   │   ├── memory_manager.py    # 3-level memory management & isolation
│   │   ├── retrieval.py         # Local vector retrieval & ranking
│   │   └── extraction.py        # Entity & fact extraction from speech
│   ├── users/
│   │   ├── user_manager.py      # Profile management & face matching
│   │   └── consent.py           # Privacy consent management
│   ├── robotics/
│   │   └── controller.py        # RobotController abstraction & telemetry
│   └── config/
│       └── settings.py          # Pydantic configuration & paths
├── dashboard/
│   ├── app.py                   # FastAPI dashboard application
│   ├── static/
│   │   ├── css/style.css        # Responsive cybernetic UI styling
│   │   └── js/app.js            # Real-time polling & audio player
│   └── templates/
│       └── index.html           # Main dashboard template
├── models/
│   ├── piper/                   # Piper TTS voice models (.onnx)
│   └── vision/                  # YuNet & SFace models (.onnx)
├── data/
│   ├── fixi.db                  # Local SQLite database
│   ├── faces/                   # Local face thumbnail crops
│   └── conversations/           # Optional conversation logs
├── tests/
│   ├── test_full_loop.py        # End-to-end Demonstration Scenario
│   ├── test_memory.py           # Strict memory isolation & 3 levels
│   ├── test_vision.py           # Person detection & face recognition
│   ├── test_speech.py           # VAD, STT, and Piper TTS synthesis
│   ├── test_users.py            # Consent flow & cascading deletion
│   └── test_robotics.py         # Odometry, pan/tilt gaze, gestures
├── requirements.txt             # Pinned local dependencies
├── pytest.ini                   # Test configuration
├── .env.example                 # Example environment variables
└── README.md                    # Project documentation
```

---

## 5. Quickstart & Installation

### Prerequisites
- Linux (x86_64) or macOS
- Python 3.11 or 3.12
- Local Ollama instance (`ollama serve`)

### Step 1: Install Dependencies
```bash
# Using uv (recommended)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Step 2: Start Local Ollama LLM
```bash
# Pull the lightweight, high-speed local model
ollama pull qwen2.5:0.5b

# Start Ollama server
ollama serve
```

### Step 3: Run the Test Suite
```bash
pytest tests/ -v
```

All 18 unit, integration, and scenario tests will run and verify every component offline.

### Step 4: Launch the Local Web Dashboard
```bash
uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
```
Open your browser to: **`http://localhost:8000`**

---

## 6. The Demonstration Scenario (Section 19)

Fixi includes an automated end-to-end test suite (`tests/test_full_loop.py`) verifying the exact vertical slice:

1. **First Interaction (Unknown Person A)**:
   - Person A approaches camera.
   - Fixi detects an unknown face and asks: *"Hello! I don't think I've met you before. Would you like me to remember you?"*
   - Person A: *"Yes."*
   - Fixi: *"Great! What is your name?"*
   - Person A: *"My name is Arun."*
   - Fixi: *"Nice to meet you, Arun!"*
   - Person A: *"I like football and I'm learning Python."*
   - Fixi extracts and stores:
     - `Name: Arun`
     - `Interest: Football`
     - `Learning: Python`

2. **Second Interaction (Arun Returns)**:
   - Arun leaves and returns later.
   - YuNet + SFace face recognition identifies Arun with high confidence (e.g. 96%).
   - Fixi greets: *"Hi Arun! Good to see you again."*
   - Arun asks: *"What do you remember about me?"*
   - Fixi answers: *"You told me that you like football and you're learning Python."*

3. **Different Person (Person B)**:
   - Person B (Maya) approaches.
   - Fixi identifies Maya and loads only Maya's memory.
   - Maya asks: *"What do you remember about me?"*
   - Fixi answers with Maya's facts only.
   - **Fixi NEVER mentions Arun's information.**

---

## 7. Robotics Hardware Integration

Fixi decouples the AI brain from physical actuators via `BaseRobotController`:

```python
from app.robotics.controller import robot_controller

# Actuator commands
robot_controller.move_forward(speed=1.0, duration=1.0)
robot_controller.turn_left(angle=45.0)
robot_controller.stop()
robot_controller.wave()

# Autonomous Gaze: automatically centers detected person in camera frame
robot_controller.look_at_person(bbox=(x, y, w, h), frame_size=(640, 480))
```

This abstraction allows plug-and-play connections to:
- ESP32 / Arduino serial bridges
- ROS 2 nodes (`cmd_vel`, servo pan/tilt)
- Raspberry Pi GPIO motor drivers
- Headless simulation (default)

---

## 8. Privacy & GDPR Compliance

- **Consent Required**: Fixi never persists memories without explicit verbal consent.
- **Local Face Embeddings**: 128-D vectors stay inside `data/fixi.db`. No cloud vector DB.
- **Right to be Forgotten**: One-click or API call `DELETE /api/users/{user_id}` permanently deletes profile, face embeddings, face images on disk, conversation history, episodic memories, and facts.
- **No Telemetry**: No tracking, phone-home, or remote metrics.

---

## 9. Point-and-Command Robotics Foundation

The project now includes a local, model-replaceable foundation for point-and-command interactions:

- `app/robotics/pointing.py`: landmark-based gesture classification, temporal pointing stability, and optional MediaPipe hand tracking.
- `app/robotics/object_detection.py`: optional local Ultralytics YOLO object detection adapter.
- `app/robotics/target_selection.py`: image-space ray scoring with confidence, persistence, and explicit ambiguity handling.
- `app/robotics/commands.py`: deterministic command parsing and identity authorization. Stop/cancel commands remain safety paths independent of the LLM.
- `app/robotics/point_command_engine.py`: point-command lifecycle and integration with the existing occupancy-grid/A* simulator.

Pointing currently maps image coordinates to the 2D simulator only. It does not claim real-world depth or 3D accuracy. MediaPipe and Ultralytics are optional runtime dependencies and must be installed locally with their model assets before camera-based point-and-command perception is available. Physical actuator movement remains disabled by the default simulated controller.
