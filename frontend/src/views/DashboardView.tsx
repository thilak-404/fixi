import React, { useState, useRef, useEffect } from 'react';
import { Camera, Mic, Volume2, User, Send, Play } from 'lucide-react';
import { PointCommandStatus, SystemStatus, WSEvent } from '../types';

interface DashboardViewProps {
  status: SystemStatus | null;
  events: WSEvent[];
  onTriggerAction: (action: string) => void;
}

export const DashboardView: React.FC<DashboardViewProps> = ({ status, events, onTriggerAction }) => {
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'fixi'; text: string; time: string }>>([
    {
      role: 'fixi',
      text: 'Hello! I am Fixi, your personal local robotic companion. I am 100% on-device and ready to assist you!',
      time: new Date().toLocaleTimeString(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [pointCommand, setPointCommand] = useState<PointCommandStatus | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const chatBottomRef = useRef<HTMLDivElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch('/api/point-command/status');
        if (res.ok) setPointCommand(await res.json());
      } catch {
        setPointCommand(null);
      }
    };
    poll();
    const interval = setInterval(poll, 500);
    return () => clearInterval(interval);
  }, []);

  const handleSendMessage = async (textToSend?: string) => {
    const text = textToSend || inputText;
    if (!text.trim() || isSending) return;

    const userMsg = { role: 'user' as const, text: text.trim(), time: new Date().toLocaleTimeString() };
    setMessages((prev) => [...prev, userMsg]);
    setInputText('');
    setIsSending(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text.trim(), user_id: status?.current_user_id || null }),
      });

      if (res.ok) {
        const data = await res.json();
        const fixiMsg = {
          role: 'fixi' as const,
          text: data.response || 'I processed that.',
          time: new Date().toLocaleTimeString(),
        };
        setMessages((prev) => [...prev, fixiMsg]);

        if (data.audio_url) {
          if (audioRef.current) {
            audioRef.current.src = data.audio_url;
            audioRef.current.play().catch(() => {});
          }
        }
      }
    } catch (e) {
      console.error('Chat request failed', e);
    } finally {
      setIsSending(false);
    }
  };

  const handleVoiceListen = async () => {
    if (isListening) {
      recorderRef.current?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      audioChunksRef.current = [];
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        setIsListening(false);
        const form = new FormData();
        form.append('audio_file', new Blob(audioChunksRef.current, { type: 'audio/webm' }), 'dashboard-command.webm');
        if (status?.current_user_id && status.current_user_id !== 'None') form.append('user_id', status.current_user_id);
        const res = await fetch('/api/voice_chat', { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) {
          setMessages((prev) => [...prev, { role: 'fixi', text: data.detail || 'Voice upload failed.', time: new Date().toLocaleTimeString() }]);
          return;
        }
        if (data.transcript) {
          const now = new Date().toLocaleTimeString();
          setMessages((prev) => [...prev, { role: 'user', text: data.transcript, time: now }, { role: 'fixi', text: data.response || 'I processed that.', time: now }]);
        }
        if (data.audio_url && audioRef.current) {
          audioRef.current.src = data.audio_url;
          audioRef.current.play().catch(() => {});
        }
      };
      recorder.start();
      setIsListening(true);
    } catch {
      setMessages((prev) => [...prev, { role: 'fixi', text: 'Microphone access was denied or recording is unavailable.', time: new Date().toLocaleTimeString() }]);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <audio ref={audioRef} />

      {/* Left Column: Live Camera & Current User (5 cols) */}
      <div className="lg:col-span-5 space-y-6">
        {/* Live Camera Card */}
        <div className="bg-[#111827] border border-[#1f293d] rounded-xl overflow-hidden shadow-lg">
          <div className="px-4 py-3 border-b border-[#1f293d] flex justify-between items-center bg-[#0d1424]">
            <div className="flex items-center gap-2 text-cyan-400 font-semibold text-sm">
              <Camera size={18} />
              <span>Live Vision Perception</span>
            </div>
            <span className="px-2 py-0.5 text-xs font-semibold rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              YOLOv8n Active
            </span>
          </div>

          <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
            <img
              src="/api/camera/stream"
              alt="Live Camera Feed"
              className="w-full h-full object-contain"
              onError={(e) => {
                (e.target as HTMLElement).style.display = 'none';
              }}
            />
            <div className="absolute top-2 left-2 bg-black/60 backdrop-blur-sm px-2 py-1 rounded text-xs text-slate-300 font-mono border border-white/10">
              FPS: {status?.camera_fps || 25} • Stream: /dev/video0
            </div>
            <div className="absolute bottom-2 left-2 right-2 bg-black/75 border border-cyan-400/30 px-3 py-2 text-xs font-mono text-cyan-100">
              <span className="text-cyan-400">POINT-COMMAND</span>{' '}
              {pointCommand?.message || 'Waiting for a stable pointing gesture.'}
              {pointCommand?.target_label ? ` Target: ${pointCommand.target_label}.` : ''}
            </div>
          </div>
        </div>

        {/* Current Person Badge */}
        <div className="bg-[#111827] border border-[#1f293d] rounded-xl p-5 shadow-lg">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-full bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center text-xl font-bold text-white shadow-md shadow-cyan-500/20">
              {status?.current_person && status.current_person !== 'None'
                ? status.current_person.charAt(0).toUpperCase()
                : '?'}
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-bold text-white">
                  {status?.current_person && status.current_person !== 'None'
                    ? status.current_person
                    : 'Awaiting Recognition'}
                </h3>
                <span className="text-xs px-2 py-0.5 rounded bg-blue-500/10 text-cyan-400 border border-cyan-500/20">
                  {status?.current_user_id || 'unassigned'}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Confidence: <strong className="text-slate-200">{status?.recognition_confidence || 0}%</strong> • Last seen:{' '}
                <strong className="text-slate-200">{status?.last_interaction || 0}s ago</strong>
              </p>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-[#1f293d] flex gap-2">
            <button
              onClick={() => onTriggerAction('wave')}
              className="flex-1 px-3 py-2 text-xs font-semibold bg-[#1e293b] hover:bg-[#27354f] text-slate-200 rounded-lg transition border border-slate-700/50 flex items-center justify-center gap-1.5"
            >
              👋 Wave Gesture
            </button>
            <button
              onClick={() => onTriggerAction('move_toward_me')}
              className="flex-1 px-3 py-2 text-xs font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 hover:opacity-95 text-white rounded-lg transition shadow-md shadow-cyan-500/20 flex items-center justify-center gap-1.5"
            >
              <Play size={14} /> Move Toward Me
            </button>
          </div>
        </div>

        {/* Live WebSocket Event Stream Drawer */}
        <div className="bg-[#111827] border border-[#1f293d] rounded-xl p-4 shadow-lg">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">WebSocket Event Stream</span>
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          </div>
          <div className="h-44 overflow-y-auto space-y-1.5 text-xs font-mono pr-1">
            {events.length === 0 ? (
              <div className="text-slate-500 text-center py-6">Connecting to real-time event bus...</div>
            ) : (
              events.slice(0, 30).map((ev, i) => (
                <div key={i} className="p-1.5 rounded bg-[#0d1424] border-l-2 border-cyan-400 text-slate-300">
                  <span className="text-slate-500 mr-2">{new Date(ev.timestamp).toLocaleTimeString()}</span>
                  <span className="font-semibold text-cyan-300 mr-1.5">{ev.type}:</span>
                  <span className="text-slate-400">
                    {ev.payload?.reason || ev.payload?.status || ev.payload?.action || JSON.stringify(ev.payload || {})}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Right Column: Two-Way Dialogue Chat (7 cols) */}
      <div className="lg:col-span-7 flex flex-col h-[740px] bg-[#111827] border border-[#1f293d] rounded-xl shadow-lg overflow-hidden">
        {/* Chat Header */}
        <div className="px-5 py-4 border-b border-[#1f293d] bg-[#0d1424] flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Volume2 size={20} />
            </div>
            <div>
              <h2 className="text-sm font-bold text-white">Fixi Interactive Dialogue</h2>
              <p className="text-xs text-slate-400">Personalized Local AI Memory • Faster-Whisper • Piper Neural TTS</p>
            </div>
          </div>
          <span className="text-xs text-cyan-400 bg-cyan-500/10 border border-cyan-500/20 px-2 py-1 rounded">
            {status?.llm_model || 'qwen2.5:0.5b'}
          </span>
        </div>

        {/* Message History */}
        <div className="flex-1 p-5 overflow-y-auto space-y-4">
          {messages.map((m, idx) => (
            <div key={idx} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ${
                  m.role === 'user'
                    ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-br-none'
                    : 'bg-[#1a2337] border border-[#263554] text-slate-100 rounded-bl-none'
                }`}
              >
                <p>{m.text}</p>
                <div
                  className={`text-[10px] mt-1.5 ${
                    m.role === 'user' ? 'text-blue-200' : 'text-slate-400'
                  } text-right`}
                >
                  {m.time}
                </div>
              </div>
            </div>
          ))}
          <div ref={chatBottomRef} />
        </div>

        {/* Chat Input Bar */}
        <div className="p-4 border-t border-[#1f293d] bg-[#0d1424]">
          <div className="flex gap-2">
            <button
              onClick={handleVoiceListen}
              className={`p-3 rounded-xl border transition flex items-center justify-center ${
                isListening
                  ? 'bg-rose-500 text-white border-rose-400 animate-pulse'
                  : 'bg-[#1a2337] hover:bg-[#25324d] text-cyan-400 border-slate-700/60'
              }`}
              title="Click to speak via browser microphone"
            >
              <Mic size={20} />
            </button>
            <input
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Ask Fixi: 'Move toward me', 'What games do I play?', or 'Remember I like python'..."
              className="flex-1 bg-[#151d30] border border-slate-700/60 rounded-xl px-4 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400 transition"
            />
            <button
              onClick={() => handleSendMessage()}
              disabled={isSending || !inputText.trim()}
              className="px-5 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:opacity-90 disabled:opacity-50 text-white font-semibold text-sm transition shadow-md shadow-cyan-500/20 flex items-center gap-2"
            >
              <Send size={16} />
              <span>Send</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
