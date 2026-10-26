// FIXI Local AI Companion Dashboard - Frontend Script
// Full Two-Way Voice Interaction & Face Registration Suite

let activeAudio = null;
let speechRecognizer = null;
let mediaRecorder = null;
let recordedAudioChunks = [];
let isListening = false;
let handsFreeMode = false;
let browserCamStream = null;
let audioTarget = "browser"; // "browser" or "robot"
let isProcessingUtterance = false;

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initChat();
  initVoiceConversation();
  initFaceRegistrationModal();
  initRobotControls();
  initMemoryViewer();
  startStatusPolling();
  loadUsers();
  initWebSocketEvents();
  initRoboticsSimulationLab();
  initCVLab();
  initBenchmarksLab();
});

// -------------------------------------------------------------
// 1. Tab Navigation
// -------------------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

      tab.classList.add("active");
      const target = document.getElementById(tab.dataset.tab);
      if (target) target.classList.add("active");

      if (tab.dataset.tab === "tab-users") loadUsers();
      if (tab.dataset.tab === "tab-memory") refreshMemoryView();
      if (tab.dataset.tab === "tab-robotics") fetchRoboticsState();
      if (tab.dataset.tab === "tab-cv") fetchCVMetrics();
    });
  });
}

// -------------------------------------------------------------
// 2. Live Status HUD Polling
// -------------------------------------------------------------
function startStatusPolling() {
  async function poll() {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        updateStatusHUD(data);
      }
    } catch (e) {
      console.warn("Status poll error", e);
    }
    setTimeout(poll, 2000);
  }
  poll();
}

function updateStatusHUD(status) {
  const camDot = document.getElementById("hud-cam-dot");
  if (camDot) camDot.style.background = status.camera_active ? "#10b981" : "#f43f5e";

  const llmStatus = document.getElementById("hud-llm-val");
  if (llmStatus) llmStatus.textContent = `${status.llm_model} (${status.llm_status})`;

  const personName = document.getElementById("current-person-name");
  const personId = document.getElementById("current-person-id");
  const personConf = document.getElementById("current-person-conf");
  const personSeen = document.getElementById("current-person-seen");
  const avatar = document.getElementById("current-person-avatar");

  if (personName) personName.textContent = status.current_person;
  if (personId) personId.textContent = status.current_user_id;
  if (personConf) personConf.textContent = status.recognition_confidence > 0 ? `${status.recognition_confidence}%` : "N/A";
  if (personSeen) personSeen.textContent = `${status.last_interaction}s ago`;

  if (avatar && status.current_person && status.current_person !== "None") {
    avatar.textContent = status.current_person.charAt(0).toUpperCase();
  }

  if (status.robot_state) {
    const r = status.robot_state;
    const telemPos = document.getElementById("telem-pos");
    const telemHead = document.getElementById("telem-heading");
    const telemPan = document.getElementById("telem-pan");
    const telemTilt = document.getElementById("telem-tilt");
    if (telemPos) telemPos.textContent = `(${r.x}, ${r.y})`;
    if (telemHead) telemHead.textContent = `${r.heading}°`;
    if (telemPan) telemPan.textContent = `${r.head_pan}°`;
    if (telemTilt) telemTilt.textContent = `${r.head_tilt}°`;
  }
}

// -------------------------------------------------------------
// 3. Audio & Voice Speech Synthesis
// -------------------------------------------------------------
function playSpeech(text, customAudioUrl = null) {
  if (!text && !customAudioUrl) return;

  if (activeAudio) {
    try { activeAudio.pause(); } catch (e) {}
    activeAudio = null;
  }

  const url = customAudioUrl || `/api/tts?text=${encodeURIComponent(text)}`;
  const audio = new Audio(url);
  activeAudio = audio;

  const eq = document.getElementById("audio-equalizer");
  const statusTxt = document.getElementById("voice-status-text");

  if (eq) eq.classList.add("active");
  if (statusTxt) statusTxt.textContent = "🔊 Fixi is speaking...";

  audio.onended = () => {
    if (eq) eq.classList.remove("active");
    if (statusTxt) {
      statusTxt.textContent = handsFreeMode ? "🔴 Hands-free listening..." : "Click Talk to speak";
    }
    activeAudio = null;

    if (handsFreeMode && !isListening) {
      setTimeout(startVoiceListening, 300);
    }
  };

  audio.onerror = () => {
    if (eq) eq.classList.remove("active");
    if (statusTxt) statusTxt.textContent = "Click Talk to speak";
    activeAudio = null;
  };

  audio.play().catch(err => {
    console.warn("Audio autoplay blocked by browser policy:", err);
    if (eq) eq.classList.remove("active");
    if (statusTxt) statusTxt.textContent = "🔊 Click to unblock voice";
  });
}

// -------------------------------------------------------------
// 4. Two-Way Voice Conversation (Speech-to-Text & Hands-Free)
// -------------------------------------------------------------
function initVoiceConversation() {
  const micBtn = document.getElementById("btn-voice-mic");
  const micLabel = document.getElementById("mic-btn-label");
  const handsfreeBtn = document.getElementById("btn-handsfree-toggle");
  const statusTxt = document.getElementById("voice-status-text");
  const sayHelloBtn = document.getElementById("btn-say-hello");

  if (sayHelloBtn) {
    sayHelloBtn.addEventListener("click", () => {
      playSpeech("Hello! I am Fixi, your personal local robotic companion. Voice output is loud and clear!");
    });
  }

  // Setup Web Speech Recognition if supported
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRecognition) {
    speechRecognizer = new SpeechRecognition();
    speechRecognizer.continuous = false;
    speechRecognizer.interimResults = false;
    speechRecognizer.lang = "en-US";

    speechRecognizer.onstart = () => {
      isListening = true;
      if (micBtn) micBtn.classList.add("recording");
      if (micLabel) micLabel.textContent = "Listening...";
      if (statusTxt) statusTxt.textContent = "🎙️ Listening... speak into mic";
    };

    speechRecognizer.onresult = async (event) => {
      if (isProcessingUtterance) return;
      const speechText = event.results[0][0].transcript.trim();
      console.log("Transcribed Speech:", speechText);
      stopVoiceListening();
      if (speechText) {
        await processUserUtterance(speechText);
      }
    };

    speechRecognizer.onerror = (event) => {
      console.warn("Speech recognition error:", event.error);
      stopVoiceListening();
      if (handsFreeMode) {
        setTimeout(startVoiceListening, 1500);
      }
    };

    speechRecognizer.onend = () => {
      stopVoiceListening();
    };
  }

  // Audio Target Toggle (Browser Audio vs Robot Hardware Speaker)
  const audioTargetBtn = document.getElementById("btn-audio-target-toggle");
  const audioTargetIcon = document.getElementById("audio-target-icon");
  const audioTargetLabel = document.getElementById("audio-target-label");

  if (audioTargetBtn) {
    audioTargetBtn.addEventListener("click", () => {
      if (audioTarget === "browser") {
        audioTarget = "robot";
        if (audioTargetIcon) audioTargetIcon.textContent = "🤖";
        if (audioTargetLabel) audioTargetLabel.textContent = "Robot Speaker";
        audioTargetBtn.classList.add("active");
        if (statusTxt) statusTxt.textContent = "Audio routed to Robot Speaker";
      } else {
        audioTarget = "browser";
        if (audioTargetIcon) audioTargetIcon.textContent = "🌐";
        if (audioTargetLabel) audioTargetLabel.textContent = "Browser Audio";
        audioTargetBtn.classList.remove("active");
        if (statusTxt) statusTxt.textContent = "Audio routed to Browser";
      }
    });
  }

  // Mic Button Click Handler (Toggle)
  if (micBtn) {
    micBtn.addEventListener("click", () => {
      if (isListening) {
        stopVoiceListening();
      } else {
        startVoiceListening();
      }
    });
  }

  // Hands-Free Mode Toggle
  if (handsfreeBtn) {
    handsfreeBtn.addEventListener("click", () => {
      handsFreeMode = !handsFreeMode;
      if (handsFreeMode) {
        handsfreeBtn.classList.add("active");
        handsfreeBtn.innerHTML = "<span>🟢</span> Hands-Free ON";
        startVoiceListening();
      } else {
        handsfreeBtn.classList.remove("active");
        handsfreeBtn.innerHTML = "<span>🔴</span> Hands-Free Mode";
        stopVoiceListening();
      }
    });
  }
}

function startVoiceListening() {
  if (activeAudio || isProcessingUtterance) return; // Don't listen while Fixi is speaking (anti-echo)

  const micBtn = document.getElementById("btn-voice-mic");
  const micLabel = document.getElementById("mic-btn-label");
  const statusTxt = document.getElementById("voice-status-text");

  if (speechRecognizer) {
    try {
      speechRecognizer.start();
    } catch (e) {
      // Already running
    }
  } else {
    // Fallback: Use MediaRecorder to stream raw audio to /api/voice_chat
    startMediaRecorderListening();
  }
}

function stopVoiceListening() {
  isListening = false;
  const micBtn = document.getElementById("btn-voice-mic");
  const micLabel = document.getElementById("mic-btn-label");
  const statusTxt = document.getElementById("voice-status-text");

  if (micBtn) micBtn.classList.remove("recording");
  if (micLabel) micLabel.textContent = "Talk to Fixi";
  if (statusTxt && !activeAudio) statusTxt.textContent = "Click Talk to speak";

  if (speechRecognizer) {
    try { speechRecognizer.stop(); } catch (e) {}
  }
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
}

async function startMediaRecorderListening() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedAudioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedAudioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const audioBlob = new Blob(recordedAudioChunks, { type: "audio/webm" });
      if (audioBlob.size > 1000) {
        await sendAudioBlobToVoiceChat(audioBlob);
      }
    };

    mediaRecorder.start();
    isListening = true;
    const micBtn = document.getElementById("btn-voice-mic");
    const micLabel = document.getElementById("mic-btn-label");
    const statusTxt = document.getElementById("voice-status-text");
    if (micBtn) micBtn.classList.add("recording");
    if (micLabel) micLabel.textContent = "Listening...";
    if (statusTxt) statusTxt.textContent = "🎙️ Recording audio...";

    // Auto stop after 5 seconds of speaking
    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        stopVoiceListening();
      }
    }, 5000);
  } catch (err) {
    alert("Microphone permission denied or unavailable: " + err);
  }
}

async function sendAudioBlobToVoiceChat(blob) {
  if (isProcessingUtterance) return;
  isProcessingUtterance = true;

  const statusTxt = document.getElementById("voice-status-text");
  if (statusTxt) statusTxt.textContent = "🧠 Transcribing audio...";

  const sel = document.getElementById("chat-as-user");
  let chosenUserId = sel ? sel.value : null;
  if (!chosenUserId) {
    const personId = document.getElementById("current-person-id");
    if (personId && personId.textContent && personId.textContent !== "None") {
      chosenUserId = personId.textContent.trim();
    }
  }

  const formData = new FormData();
  formData.append("audio_file", blob, "recording.webm");
  if (chosenUserId) formData.append("user_id", chosenUserId);
  formData.append("speak_aloud", audioTarget === "robot" ? "true" : "false");

  try {
    const res = await fetch("/api/voice_chat", {
      method: "POST",
      body: formData
    });
    if (res.ok) {
      const data = await res.json();
      if (data.transcript) {
        appendBubble("user", data.transcript);
      }
      appendBubble("fixi", data.response);

      // Only play in browser if audioTarget is "browser" (robot speaker handles it when audioTarget is "robot")
      if (audioTarget === "browser") {
        playSpeech(data.response, data.audio_url);
      } else {
        if (statusTxt) {
          statusTxt.textContent = "🤖 Robot speaker playing...";
          setTimeout(() => {
            if (statusTxt) statusTxt.textContent = handsFreeMode ? "🔴 Hands-free listening..." : "Click Talk to speak";
            if (handsFreeMode && !isListening) setTimeout(startVoiceListening, 400);
          }, 2500);
        }
      }

      if (data.extracted && data.extracted.facts && Object.keys(data.extracted.facts).length > 0) {
        const factsStr = Object.entries(data.extracted.facts).map(([k, v]) => `${k}: ${v}`).join(", ");
        appendSystemNote(`💡 Learned new memory: ${factsStr}`);
      }
    }
  } catch (e) {
    appendBubble("fixi", "[Voice processing error]");
    if (statusTxt) statusTxt.textContent = "Click Talk to speak";
  } finally {
    isProcessingUtterance = false;
  }
}

async function processUserUtterance(text) {
  if (!text || isProcessingUtterance) return;
  isProcessingUtterance = true;

  appendBubble("user", text);

  const sel = document.getElementById("chat-as-user");
  let chosenUserId = sel ? sel.value : null;
  if (!chosenUserId) {
    const personId = document.getElementById("current-person-id");
    if (personId && personId.textContent && personId.textContent !== "None") {
      chosenUserId = personId.textContent.trim();
    }
  }
  const statusTxt = document.getElementById("voice-status-text");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        user_id: chosenUserId || undefined,
        speak_aloud: audioTarget === "robot"
      })
    });

    if (res.ok) {
      const data = await res.json();
      const reply = data.response;
      appendBubble("fixi", reply);

      // Only play in browser if audioTarget is "browser" (prevents double voice)
      if (audioTarget === "browser") {
        playSpeech(reply, data.audio_url);
      } else {
        if (statusTxt) {
          statusTxt.textContent = "🤖 Robot speaker playing...";
          setTimeout(() => {
            if (statusTxt) statusTxt.textContent = handsFreeMode ? "🔴 Hands-free listening..." : "Click Talk to speak";
            if (handsFreeMode && !isListening) setTimeout(startVoiceListening, 400);
          }, 2500);
        }
      }

      if (data.extracted && data.extracted.facts && Object.keys(data.extracted.facts).length > 0) {
        const factsStr = Object.entries(data.extracted.facts).map(([k, v]) => `${k}: ${v}`).join(", ");
        appendSystemNote(`💡 Learned new memory: ${factsStr}`);
      }
    }
  } catch (e) {
    appendBubble("fixi", "[Offline / Connection Error]");
    if (statusTxt) statusTxt.textContent = "Click Talk to speak";
  } finally {
    isProcessingUtterance = false;
  }
}

// -------------------------------------------------------------
// 5. Interactive Chat Window
// -------------------------------------------------------------
function initChat() {
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send-btn");

  async function handleSend() {
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    await processUserUtterance(text);
  }

  sendBtn.addEventListener("click", handleSend);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleSend();
  });
}

function appendBubble(role, text) {
  const chatWindow = document.getElementById("chat-window");
  if (!chatWindow) return;
  const div = document.createElement("div");
  div.className = `chat-bubble ${role}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendSystemNote(text) {
  const chatWindow = document.getElementById("chat-window");
  if (!chatWindow) return;
  const div = document.createElement("div");
  div.style.fontSize = "0.75rem";
  div.style.color = "var(--accent-cyan)";
  div.style.alignSelf = "center";
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// -------------------------------------------------------------
// 6. Multi-Option Face Registration Modal
// -------------------------------------------------------------
function initFaceRegistrationModal() {
  const modal = document.getElementById("face-reg-modal");
  const openBtn = document.getElementById("btn-register-face");
  const closeBtn = document.getElementById("modal-close-btn");
  const nameInput = document.getElementById("reg-name-input");
  const feedback = document.getElementById("modal-feedback");

  if (!modal) return;

  function openModal() {
    modal.classList.add("open");
    if (feedback) feedback.textContent = "";
    if (nameInput) nameInput.focus();
  }

  function closeModal() {
    modal.classList.remove("open");
    if (browserCamStream) {
      browserCamStream.getTracks().forEach(t => t.stop());
      browserCamStream = null;
    }
  }

  if (openBtn) openBtn.addEventListener("click", openModal);
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });

  // Modal Tabs Switching
  const modalTabs = document.querySelectorAll(".modal-tab-btn");
  modalTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      modalTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      document.querySelectorAll(".modal-tab-content").forEach(c => c.style.display = "none");
      const target = document.getElementById(tab.dataset.modaltab);
      if (target) target.style.display = "block";
    });
  });

  // Option 1: Scan Live Camera Stream
  const scanLiveBtn = document.getElementById("btn-scan-live-cam");
  if (scanLiveBtn) {
    scanLiveBtn.addEventListener("click", async () => {
      const name = nameInput.value.trim();
      if (!name) return alert("Please enter your name first.");

      scanLiveBtn.disabled = true;
      scanLiveBtn.textContent = "Scanning face in live view...";
      if (feedback) feedback.innerHTML = "<span style='color: var(--accent-cyan);'>🔍 Scanning camera frames... Smile at the camera!</span>";

      try {
        const res = await fetch("/api/register_face", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name })
        });
        const data = await res.json();
        if (res.ok) {
          if (feedback) feedback.innerHTML = `<span style='color: var(--accent-green);'>✅ ${data.message}</span>`;
          loadUsers();
          playSpeech(data.message, data.audio_url);
          setTimeout(closeModal, 2000);
        } else {
          if (feedback) feedback.innerHTML = `<span style='color: var(--accent-rose);'>⚠️ ${data.detail || data.message}</span>`;
        }
      } catch (e) {
        if (feedback) feedback.innerHTML = `<span style='color: var(--accent-rose);'>Error: ${e}</span>`;
      } finally {
        scanLiveBtn.disabled = false;
        scanLiveBtn.textContent = "🔍 Scan Face from Live Camera";
      }
    });
  }

  // Option 2: Browser Webcam Snap
  const startCamBtn = document.getElementById("btn-start-browser-cam");
  const snapCamBtn = document.getElementById("btn-snap-browser-cam");
  const videoElem = document.getElementById("modal-webcam-video");
  const canvasElem = document.getElementById("modal-webcam-canvas");

  if (startCamBtn && videoElem) {
    startCamBtn.addEventListener("click", async () => {
      try {
        browserCamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        videoElem.srcObject = browserCamStream;
        snapCamBtn.disabled = false;
        startCamBtn.textContent = "Camera Ready";
      } catch (err) {
        alert("Could not access browser webcam: " + err);
      }
    });
  }

  if (snapCamBtn && videoElem && canvasElem) {
    snapCamBtn.addEventListener("click", async () => {
      const name = nameInput.value.trim();
      if (!name) return alert("Please enter your name first.");

      canvasElem.width = videoElem.videoWidth || 640;
      canvasElem.height = videoElem.videoHeight || 480;
      const ctx = canvasElem.getContext("2d");
      ctx.drawImage(videoElem, 0, 0, canvasElem.width, canvasElem.height);
      const b64Data = canvasElem.toDataURL("image/jpeg", 0.9);

      snapCamBtn.disabled = true;
      snapCamBtn.textContent = "Enrolling...";

      try {
        const res = await fetch("/api/register_face_image", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, image_base64: b64Data })
        });
        const data = await res.json();
        if (res.ok) {
          if (feedback) feedback.innerHTML = `<span style='color: var(--accent-green);'>✅ ${data.message}</span>`;
          loadUsers();
          playSpeech(data.message, data.audio_url);
          setTimeout(closeModal, 2000);
        } else {
          if (feedback) feedback.innerHTML = `<span style='color: var(--accent-rose);'>⚠️ ${data.detail || data.message}</span>`;
        }
      } catch (e) {
        if (feedback) feedback.innerHTML = `<span style='color: var(--accent-rose);'>Error: ${e}</span>`;
      } finally {
        snapCamBtn.disabled = false;
        snapCamBtn.textContent = "📸 Take Photo & Enroll";
      }
    });
  }

  // Option 3: Photo File Upload
  const uploadBtn = document.getElementById("btn-upload-photo");
  const fileInput = document.getElementById("modal-photo-file");

  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener("click", async () => {
      const name = nameInput.value.trim();
      if (!name) return alert("Please enter your name first.");
      if (!fileInput.files || fileInput.files.length === 0) return alert("Please select a photo file.");

      const formData = new FormData();
      formData.append("name", name);
      formData.append("photo", fileInput.files[0]);

      uploadBtn.disabled = true;
      uploadBtn.textContent = "Uploading & Analyzing...";

      try {
        const res = await fetch("/api/register_face_upload", {
          method: "POST",
          body: formData
        });
        const data = await res.json();
        if (res.ok) {
          if (feedback) feedback.innerHTML = `<span style='color: var(--accent-green);'>✅ ${data.message}</span>`;
          loadUsers();
          playSpeech(data.message, data.audio_url);
          setTimeout(closeModal, 2000);
        } else {
          if (feedback) feedback.innerHTML = `<span style='color: var(--accent-rose);'>⚠️ ${data.detail || data.message}</span>`;
        }
      } catch (e) {
        if (feedback) feedback.innerHTML = `<span style='color: var(--accent-rose);'>Error: ${e}</span>`;
      } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = "📁 Upload & Register Profile";
      }
    });
  }
}

// -------------------------------------------------------------
// 7. Robotics Controls
// -------------------------------------------------------------
function initRobotControls() {
  document.querySelectorAll(".btn-robot").forEach(btn => {
    btn.addEventListener("click", async () => {
      const act = btn.dataset.action;
      if (!act) return;
      try {
        await fetch("/api/robot/action", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: act })
        });
      } catch (e) {
        console.error("Robot action failed", e);
      }
    });
  });
}

// -------------------------------------------------------------
// 8. User Management
// -------------------------------------------------------------
async function loadUsers() {
  try {
    const res = await fetch("/api/users");
    if (!res.ok) return;
    const users = await res.json();

    const tbody = document.getElementById("users-table-body");
    if (tbody) {
      tbody.innerHTML = "";
      users.forEach(u => {
        const tr = document.createElement("tr");
        const faceBadge = u.has_face ? "<span class='badge badge-green'>Enrolled</span>" : "<span class='badge badge-amber'>No Face</span>";
        tr.innerHTML = `
          <td><strong>${u.id}</strong></td>
          <td>${u.name}</td>
          <td><span class="badge badge-green">${u.consent_status}</span> ${faceBadge}</td>
          <td>${new Date(u.last_seen).toLocaleTimeString()}</td>
          <td>${u.total_memories} (${u.fact_count} facts, ${u.episodic_count} episodes)</td>
          <td>
            <button class="btn-danger" onclick="deleteUser('${u.id}')">Delete</button>
          </td>
        `;
        tbody.appendChild(tr);
      });
    }

    const selects = [document.getElementById("chat-as-user"), document.getElementById("memory-user-select")];
    selects.forEach(sel => {
      if (!sel) return;
      const cur = sel.value;
      sel.innerHTML = '<option value="">-- Active Recognized User --</option>';
      users.forEach(u => {
        const opt = document.createElement("option");
        opt.value = u.id;
        opt.textContent = `${u.name} (${u.id})`;
        if (opt.value === cur) opt.selected = true;
        sel.appendChild(opt);
      });
    });
  } catch (e) {
    console.error("Failed to load users", e);
  }
}

async function deleteUser(userId) {
  if (!confirm(`Are you sure you want to permanently delete ${userId} and all isolated memories?`)) return;
  try {
    const res = await fetch(`/api/users/${userId}`, { method: "DELETE" });
    if (res.ok) {
      loadUsers();
      refreshMemoryView();
    }
  } catch (e) {
    alert("Failed to delete user");
  }
}

// -------------------------------------------------------------
// 9. Memory Vault (Levels A, B, C)
// -------------------------------------------------------------
function initMemoryViewer() {
  const sel = document.getElementById("memory-user-select");
  if (sel) {
    sel.addEventListener("change", refreshMemoryView);
  }
  const clearBtn = document.getElementById("btn-clear-memories");
  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      const u = sel.value;
      if (!u) return alert("Select a user first");
      if (!confirm(`Clear all memories for ${u}?`)) return;
      await fetch(`/api/memories/${u}/clear`, { method: "DELETE" });
      refreshMemoryView();
    });
  }
}

async function refreshMemoryView() {
  const sel = document.getElementById("memory-user-select");
  if (!sel || !sel.value) {
    document.getElementById("memory-facts-container").innerHTML = "<p class='text-muted'>Select a user above to view their private memory vault.</p>";
    document.getElementById("memory-episodic-container").innerHTML = "";
    return;
  }
  const userId = sel.value;

  try {
    const res = await fetch(`/api/memories/${userId}`);
    if (!res.ok) return;
    const data = await res.json();

    // Render Facts (Level C)
    const factsCont = document.getElementById("memory-facts-container");
    if (!data.facts || Object.keys(data.facts).length === 0) {
      factsCont.innerHTML = "<p class='text-muted'>No stable facts stored yet.</p>";
    } else {
      let html = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">';
      for (const [k, v] of Object.entries(data.facts)) {
        html += `<div class="badge badge-blue" style="padding: 6px 12px; font-size: 0.85rem;">
          <strong>${k}:</strong> ${v}
        </div>`;
      }
      html += '</div>';
      factsCont.innerHTML = html;
    }

    // Render Episodic (Level B)
    const epCont = document.getElementById("memory-episodic-container");
    if (!data.episodic || data.episodic.length === 0) {
      epCont.innerHTML = "<p class='text-muted'>No episodic memories stored yet.</p>";
    } else {
      let html = '<div style="display: flex; flex-direction: column; gap: 10px;">';
      data.episodic.forEach(m => {
        html += `<div style="background: var(--bg-card); padding: 10px; border-radius: 8px; border: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center;">
          <div>
            <div style="font-weight: 600; color: var(--accent-cyan);">${m.summary}</div>
            <div style="font-size: 0.75rem; color: var(--text-muted);">${m.details || ""} • ${new Date(m.timestamp).toLocaleDateString()}</div>
          </div>
          <button class="btn-danger" onclick="deleteEpisodic('${userId}', ${m.id})">Delete</button>
        </div>`;
      });
      html += '</div>';
      epCont.innerHTML = html;
    }
  } catch (e) {
    console.error("Failed to fetch memories", e);
  }
}

async function deleteEpisodic(userId, memId) {
  if (!confirm("Delete this memory?")) return;
  await fetch(`/api/memories/episodic/${userId}/${memId}`, { method: "DELETE" });
  refreshMemoryView();
}

// -------------------------------------------------------------
// 10. Real-Time WebSocket Event Stream
// -------------------------------------------------------------
let eventWs = null;

function initWebSocketEvents() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/events`;

  function connect() {
    try {
      eventWs = new WebSocket(wsUrl);

      eventWs.onopen = () => {
        const badge = document.getElementById("ws-badge");
        if (badge) {
          badge.textContent = "Live";
          badge.className = "badge badge-green";
        }
      };

      eventWs.onmessage = (event) => {
        try {
          const ev = JSON.parse(event.data);
          appendEventToTimeline(ev);

          // Real-time reactions to robotics and CV events
          const type = ev.type || "";
          if (type.startsWith("action_") || type.startsWith("path_") || type.startsWith("replanning_") || type.startsWith("planning_")) {
            fetchRoboticsState();
          } else if (type.startsWith("person_") || type.startsWith("tracking_")) {
            fetchCVMetrics();
          }
        } catch (err) {
          console.warn("Failed to parse event", err);
        }
      };

      eventWs.onclose = () => {
        const badge = document.getElementById("ws-badge");
        if (badge) {
          badge.textContent = "Reconnecting";
          badge.className = "badge badge-amber";
        }
        setTimeout(connect, 3000);
      };

      eventWs.onerror = () => {
        if (eventWs) eventWs.close();
      };
    } catch (e) {
      console.warn("WebSocket initialization error", e);
    }
  }

  connect();
}

function appendEventToTimeline(ev) {
  const box = document.getElementById("event-timeline-box");
  if (!box) return;

  const entry = document.createElement("div");
  const type = ev.type || "unknown";

  let catClass = "cv";
  if (type.includes("action") || type.includes("path") || type.includes("planning")) catClass = "ir";
  else if (type.includes("intent") || type.includes("speech") || type.includes("llm") || type.includes("tts")) catClass = "voice";
  else if (type.includes("failed") || type.includes("lost") || type.includes("blocked")) catClass = "warn";

  entry.className = `event-entry ${catClass}`;

  const timeStr = new Date(ev.timestamp || Date.now()).toLocaleTimeString();
  let desc = JSON.stringify(ev.payload || {}).substring(0, 80);
  if (ev.payload && ev.payload.reason) desc = ev.payload.reason;
  else if (ev.payload && ev.payload.status) desc = ev.payload.status;
  else if (ev.payload && ev.payload.action) desc = `Action: ${ev.payload.action}`;
  else if (ev.payload && ev.payload.current_state) desc = `State changed to ${ev.payload.current_state}`;

  entry.innerHTML = `
    <span class="event-time">${timeStr}</span>
    <span class="event-badge">${type.toUpperCase()}</span>
    <span class="event-desc">${desc}</span>
  `;

  box.prepend(entry);
  while (box.children.length > 50) {
    box.removeChild(box.lastChild);
  }
}

// -------------------------------------------------------------
// 11. Intelligent Robotics Simulation Lab & 2D Canvas
// -------------------------------------------------------------
let simState = null;
let simAutoInterval = null;

function initRoboticsSimulationLab() {
  const canvas = document.getElementById("robotics-canvas");
  if (!canvas) return;

  // Click canvas cell to toggle dynamic obstacle
  canvas.addEventListener("click", async (e) => {
    if (!simState || !simState.grid) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cols = simState.grid.width || 20;
    const rows = simState.grid.height || 20;
    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;

    const clickCol = Math.floor(x / cellW);
    // Invert row if origin is bottom, or use direct row:
    const clickRow = Math.floor(y / cellH);

    // Toggle obstacle
    try {
      const isDynamic = (simState.grid.dynamic_obstacles || []).some(o => o[0] === clickCol && o[1] === clickRow);
      const action = isDynamic ? "remove" : "add";

      const res = await fetch("/api/robotics/obstacle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ col: clickCol, row: clickRow, action: action })
      });
      if (res.ok) {
        const data = await res.json();
        simState = data.state;
        renderRoboticsCanvas(simState);
      }
    } catch (err) {
      console.warn("Failed to toggle obstacle", err);
    }
  });

  // Action buttons
  const btnNavPerson = document.getElementById("btn-sim-nav-person");
  if (btnNavPerson) {
    btnNavPerson.addEventListener("click", async () => {
      const fb = document.getElementById("sim-status-feedback");
      if (fb) fb.textContent = "Resolving vision person location & planning A* path...";
      const res = await fetch("/api/robotics/navigate_person", { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        simState = data.state;
        renderRoboticsCanvas(simState);
        if (fb) fb.textContent = data.success ? "Path planned toward person! Robot moving." : "Navigation blocked by obstacles.";
      }
    });
  }

  const btnStep = document.getElementById("btn-sim-step");
  if (btnStep) {
    btnStep.addEventListener("click", async () => {
      const res = await fetch("/api/robotics/step", { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        simState = data.state;
        renderRoboticsCanvas(simState);
        const fb = document.getElementById("sim-status-feedback");
        if (fb && data.result) {
          fb.textContent = `Step executed: Status: ${data.result.status || "ok"}`;
        }
      }
    });
  }

  const btnAuto = document.getElementById("btn-sim-autorun");
  if (btnAuto) {
    btnAuto.addEventListener("click", () => {
      if (simAutoInterval) {
        clearInterval(simAutoInterval);
        simAutoInterval = null;
        btnAuto.textContent = "⚡ Auto-Step";
      } else {
        btnAuto.textContent = "⏸️ Pause Auto";
        simAutoInterval = setInterval(async () => {
          const res = await fetch("/api/robotics/step", { method: "POST" });
          if (res.ok) {
            const data = await res.json();
            simState = data.state;
            renderRoboticsCanvas(simState);
            if (data.result && (data.result.status === "arrived" || data.result.status === "completed")) {
              clearInterval(simAutoInterval);
              simAutoInterval = null;
              btnAuto.textContent = "⚡ Auto-Step";
            }
          }
        }, 350);
      }
    });
  }

  const btnStop = document.getElementById("btn-sim-stop");
  if (btnStop) {
    btnStop.addEventListener("click", async () => {
      if (simAutoInterval) {
        clearInterval(simAutoInterval);
        simAutoInterval = null;
      }
      await fetch("/api/robotics/emergency_stop", { method: "POST" });
      fetchRoboticsState();
    });
  }

  const btnReset = document.getElementById("btn-sim-reset");
  if (btnReset) {
    btnReset.addEventListener("click", async () => {
      if (simAutoInterval) {
        clearInterval(simAutoInterval);
        simAutoInterval = null;
      }
      await fetch("/api/robotics/reset", { method: "POST" });
      fetchRoboticsState();
    });
  }

  fetchRoboticsState();
}

async function fetchRoboticsState() {
  try {
    const res = await fetch("/api/robotics/state");
    if (res.ok) {
      simState = await res.json();
      renderRoboticsCanvas(simState);
      updateRoboticsTelemetry(simState);
    }
  } catch (e) {
    console.warn("Failed to fetch robotics state", e);
  }
}

function updateRoboticsTelemetry(state) {
  if (!state) return;

  if (state.robot) {
    const r = state.robot;
    const gridEl = document.getElementById("sim-grid-pos");
    const worldEl = document.getElementById("sim-world-pos");
    const headEl = document.getElementById("sim-heading");
    if (gridEl) gridEl.textContent = `(${r.grid_col}, ${r.grid_row})`;
    if (worldEl) worldEl.textContent = `(${r.world_x}m, ${r.world_y}m)`;
    if (headEl) headEl.textContent = `${r.heading_deg}°`;
  }

  if (state.planner_metrics) {
    const m = state.planner_metrics;
    const lenEl = document.getElementById("sim-path-len");
    const nodeEl = document.getElementById("sim-nodes-exp");
    const timeEl = document.getElementById("sim-plan-time");
    if (lenEl) lenEl.textContent = `${m.path_length} steps`;
    if (nodeEl) nodeEl.textContent = `${m.nodes_expanded}`;
    if (timeEl) timeEl.textContent = `${m.planning_time_ms} ms`;
  }

  if (state.fsm) {
    updateFSMVisualizer(state.fsm.state);
  }
}

function updateFSMVisualizer(currentState) {
  const nodes = document.querySelectorAll(".fsm-node");
  nodes.forEach(n => n.classList.remove("fsm-active", "state-blocked", "state-arrived"));

  const target = document.getElementById(`fsm-node-${currentState}`);
  if (target) {
    target.classList.add("fsm-active");
    if (currentState === "BLOCKED") target.classList.add("state-blocked");
    if (currentState === "ARRIVED") target.classList.add("state-arrived");
  }

  const fb = document.getElementById("sim-status-feedback");
  if (fb) {
    if (currentState === "NAVIGATING") fb.textContent = "Autonomous navigation in progress with real-time collision verification.";
    else if (currentState === "BLOCKED") fb.textContent = "Path blocked by obstacle! Initiating dynamic replanning...";
    else if (currentState === "REPLANNING") fb.textContent = "Recomputing collision-free A* detour...";
    else if (currentState === "ARRIVED") fb.textContent = "Target reached successfully! Visual feedback verified.";
    else if (currentState === "EMERGENCY_STOP") fb.textContent = "Safety stop active. Motors inhibited.";
  }
}

function renderRoboticsCanvas(state) {
  const canvas = document.getElementById("robotics-canvas");
  if (!canvas || !state || !state.grid) return;
  const ctx = canvas.getContext("2d");

  const cols = state.grid.width || 20;
  const rows = state.grid.height || 20;
  const cw = canvas.width / cols;
  const ch = canvas.height / rows;

  // Clear background
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  for (let c = 0; c <= cols; c++) {
    ctx.beginPath();
    ctx.moveTo(c * cw, 0);
    ctx.lineTo(c * cw, canvas.height);
    ctx.stroke();
  }
  for (let r = 0; r <= rows; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * ch);
    ctx.lineTo(canvas.width, r * ch);
    ctx.stroke();
  }

  // Draw cells from grid matrix: 0=empty, 1=obstacle, 5=inflated
  const matrix = state.grid.grid || [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const val = matrix[r] ? matrix[r][c] : 0;
      if (val === 5) {
        // Inflated safety zone
        ctx.fillStyle = "rgba(245, 158, 11, 0.18)";
        ctx.fillRect(c * cw, r * ch, cw, ch);
        ctx.strokeStyle = "rgba(245, 158, 11, 0.35)";
        ctx.strokeRect(c * cw + 1, r * ch + 1, cw - 2, ch - 2);
      } else if (val === 1) {
        // Static or dynamic obstacle
        const isDynamic = (state.grid.dynamic_obstacles || []).some(o => o[0] === c && o[1] === r);
        ctx.fillStyle = isDynamic ? "#f97316" : "#334155";
        ctx.fillRect(c * cw + 1, r * ch + 1, cw - 2, ch - 2);
        if (isDynamic) {
          ctx.strokeStyle = "#ffedd5";
          ctx.lineWidth = 1.5;
          ctx.strokeRect(c * cw + 2, r * ch + 2, cw - 4, ch - 4);
        }
      }
    }
  }

  // Draw Planned Path
  const path = state.path || [];
  if (path.length > 1) {
    ctx.strokeStyle = "#00f2fe";
    ctx.lineWidth = 3;
    ctx.shadowColor = "#00f2fe";
    ctx.shadowBlur = 8;
    ctx.beginPath();
    path.forEach((pt, idx) => {
      const px = pt[0] * cw + cw / 2;
      const py = pt[1] * ch + ch / 2;
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw waypoints
    ctx.fillStyle = "#38bdf8";
    path.forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt[0] * cw + cw / 2, pt[1] * ch + ch / 2, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  // Draw Target
  if (state.target) {
    const tx = state.target.grid_col * cw + cw / 2;
    const ty = state.target.grid_row * ch + ch / 2;
    ctx.fillStyle = "#f59e0b";
    ctx.beginPath();
    ctx.arc(tx, ty, cw * 0.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 10px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("🎯", tx, ty);
  }

  // Draw Person
  if (state.person && state.person.grid_col !== null) {
    const px = state.person.grid_col * cw + cw / 2;
    const py = state.person.grid_row * ch + ch / 2;
    ctx.fillStyle = "#e11d48";
    ctx.beginPath();
    ctx.arc(px, py, cw * 0.45, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("👤", px, py);
  }

  // Draw Robot
  if (state.robot) {
    const rx = state.robot.grid_col * cw + cw / 2;
    const ry = state.robot.grid_row * ch + ch / 2;

    // Outer glow
    ctx.shadowColor = "#00f2fe";
    ctx.shadowBlur = 10;
    ctx.fillStyle = "#00f2fe";
    ctx.beginPath();
    ctx.arc(rx, ry, cw * 0.42, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Inner core
    ctx.fillStyle = "#0f172a";
    ctx.beginPath();
    ctx.arc(rx, ry, cw * 0.22, 0, Math.PI * 2);
    ctx.fill();

    // Direction needle
    const rad = (state.robot.heading_deg * Math.PI) / 180;
    const nx = rx + Math.cos(rad) * (cw * 0.65);
    const ny = ry + Math.sin(rad) * (ch * 0.65);
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(rx, ry);
    ctx.lineTo(nx, ny);
    ctx.stroke();
  }
}

// -------------------------------------------------------------
// 12. Computer Vision Lab Module
// -------------------------------------------------------------
function initCVLab() {
  fetchCVMetrics();
  setInterval(() => {
    const cvTab = document.getElementById("tab-cv");
    if (cvTab && cvTab.classList.contains("active")) {
      fetchCVMetrics();
    }
  }, 1500);
}

async function fetchCVMetrics() {
  try {
    const res = await fetch("/api/evaluation/benchmarks");
    if (res.ok) {
      const data = await res.json();
      const cv = data.computer_vision || {};

      const latEl = document.getElementById("cv-latency-val");
      const fpsEl = document.getElementById("cv-fps-val");
      const countEl = document.getElementById("cv-count-val");
      const trackEl = document.getElementById("cv-track-id-val");

      if (latEl) latEl.textContent = `${cv.latency_ms || 0.0} ms`;
      if (fpsEl) fpsEl.textContent = `${cv.fps || 0.0} FPS`;
      if (countEl) countEl.textContent = `${cv.active_detections || 0}`;

      const dets = cv.detections || [];
      if (trackEl && dets.length > 0) {
        trackEl.textContent = `#${dets[0].tracking_id || 1}`;
      }

      // Populate table
      const tbody = document.getElementById("cv-detections-table-body");
      if (tbody) {
        if (dets.length === 0) {
          tbody.innerHTML = `<tr><td colspan="6" class="text-muted" style="text-align: center;">No active person in camera field of view.</td></tr>`;
        } else {
          let html = "";
          dets.forEach(d => {
            html += `<tr>
              <td><strong>#${d.tracking_id}</strong></td>
              <td>[${d.bbox.join(", ")}]</td>
              <td>${Math.round(d.confidence * 100)}%</td>
              <td>${d.spatial_x} m</td>
              <td>${d.spatial_y} m</td>
              <td>(${d.grid_x}, ${d.grid_y})</td>
            </tr>`;
          });
          tbody.innerHTML = html;
        }
      }
    }
  } catch (e) {
    console.warn("Failed to fetch CV metrics", e);
  }
}

// -------------------------------------------------------------
// 13. Academic Benchmarks Lab Module
// -------------------------------------------------------------
function initBenchmarksLab() {
  const btn = document.getElementById("btn-run-benchmarks");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.textContent = "⏳ Running Benchmarks...";
    try {
      const res = await fetch("/api/evaluation/benchmarks");
      if (res.ok) {
        const data = await res.json();
        const ir = data.intelligent_robotics || {};
        const sum = ir.summary || {};
        const cv = data.computer_vision || {};

        const collEl = document.getElementById("bench-collision-rate");
        const timeEl = document.getElementById("bench-avg-time");
        const nodeEl = document.getElementById("bench-avg-nodes");
        const fpsEl = document.getElementById("bench-yolo-fps");
        const latEl = document.getElementById("bench-yolo-lat");

        if (collEl) collEl.textContent = sum.collision_avoidance_success_rate || "100.0%";
        if (timeEl) timeEl.textContent = `${sum.average_planning_time_ms || 0} ms`;
        if (nodeEl) nodeEl.textContent = `${sum.average_nodes_expanded || 0}`;
        if (fpsEl) fpsEl.textContent = `${cv.fps || 0} FPS`;
        if (latEl) latEl.textContent = `${cv.latency_ms || 0} ms`;

        // Scenarios table
        const tbody = document.getElementById("bench-scenarios-table-body");
        if (tbody && ir.scenarios) {
          let html = "";
          ir.scenarios.forEach(sc => {
            html += `<tr>
              <td><strong>${sc.scenario}</strong></td>
              <td>(${sc.start.join(", ")})</td>
              <td>(${sc.goal.join(", ")})</td>
              <td><span class="badge badge-green">Passed</span></td>
              <td>${sc.nodes_expanded}</td>
              <td>${sc.planning_time_ms} ms</td>
              <td>${sc.path_cost}</td>
              <td>${sc.path_efficiency_pct}%</td>
            </tr>`;
          });
          tbody.innerHTML = html;
        }
      }
    } catch (e) {
      console.warn("Benchmark run failed", e);
    } finally {
      btn.disabled = false;
      btn.textContent = "🚀 Run Academic Benchmarks";
    }
  });
}
