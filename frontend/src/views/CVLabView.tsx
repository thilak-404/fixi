import React, { useEffect, useRef, useState } from 'react';
import { Camera, Activity, Layers, Target, Eye, UserPlus, Mic, Send, ShieldCheck } from 'lucide-react';
import { PersonDetection } from '../types';

export const CVLabView: React.FC = () => {
  const [metrics, setMetrics] = useState<any>(null);
  const [detections, setDetections] = useState<PersonDetection[]>([]);
  const [name, setName] = useState('');
  const [voiceText, setVoiceText] = useState('');
  const [controlMessage, setControlMessage] = useState('');
  const [isListening, setIsListening] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    const fetchCV = async () => {
      try {
        const res = await fetch('/api/evaluation/benchmarks');
        if (res.ok) {
          const data = await res.json();
          const cv = data.computer_vision || {};
          setMetrics(cv);
          setDetections(cv.detections || []);
        }
      } catch (e) {
        console.warn('Failed to fetch CV data', e);
      }
    };

    fetchCV();
    const interval = setInterval(fetchCV, 1500);
    return () => clearInterval(interval);
  }, []);

  const registerCurrentFace = async () => {
    if (!name.trim()) {
      setControlMessage('Enter a name before registering the visible face.');
      return;
    }
    const res = await fetch('/api/register_face', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim() }),
    });
    const data = await res.json();
    setControlMessage(res.ok ? data.message : data.detail || 'Face registration failed.');
    if (res.ok) setName('');
  };

  const startSpeechRecognition = async () => {
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
        form.append('audio_file', new Blob(audioChunksRef.current, { type: 'audio/webm' }), 'cv-command.webm');
        const response = await fetch('/api/voice_chat', { method: 'POST', body: form });
        const data = await response.json();
        setVoiceText(data.transcript || '');
        setControlMessage(data.response || data.detail || 'No response received.');
      };
      recorder.start();
      setIsListening(true);
      setControlMessage('Listening... press the microphone button again to finish.');
    } catch {
      setControlMessage('Microphone access was denied or MediaRecorder is unavailable.');
    }
  };

  const sendSystemCommand = async (text = voiceText) => {
    if (!text.trim()) return;
    const res = await fetch('/api/system/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text.trim() }),
    });
    const data = await res.json();
    setControlMessage(res.ok ? data.message : data.detail || 'Command was rejected.');
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-[#111827] border border-cyan-500/30 p-5 rounded-xl shadow-lg">
          <div className="flex items-center gap-2 text-cyan-300 text-sm font-bold mb-3">
            <UserPlus size={18} /> Register Face From Live Camera
          </div>
          <p className="text-xs text-slate-400 mb-3">Keep one face centered in the camera. Registration stores the local embedding only after you provide a name.</p>
          <div className="flex gap-2">
            <input value={name} onChange={(event) => setName(event.target.value)} placeholder="Name" className="flex-1 bg-[#151d30] border border-slate-700 rounded-lg px-3 py-2 text-sm" />
            <button onClick={registerCurrentFace} className="px-3 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold flex items-center gap-1.5"><ShieldCheck size={15} /> Register</button>
          </div>
        </div>
        <div className="bg-[#111827] border border-emerald-500/30 p-5 rounded-xl shadow-lg">
          <div className="flex items-center gap-2 text-emerald-300 text-sm font-bold mb-3">
            <Mic size={18} /> Local Speech Controls
          </div>
          <p className="text-xs text-slate-400 mb-3">Examples: “open browser”, “open terminal”, or “remind me to drink water in 10 minutes”. Commands require the recognized active user.</p>
          <div className="flex gap-2">
            <button onClick={startSpeechRecognition} className={`px-3 py-2 rounded-lg text-xs font-bold ${isListening ? 'bg-rose-600 text-white animate-pulse' : 'bg-emerald-600 hover:bg-emerald-500 text-white'}`}><Mic size={15} /></button>
            <input value={voiceText} onChange={(event) => setVoiceText(event.target.value)} placeholder="Speak or type a local command" className="flex-1 bg-[#151d30] border border-slate-700 rounded-lg px-3 py-2 text-sm" />
            <button onClick={() => void sendSystemCommand()} className="px-3 py-2 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-white"><Send size={15} /></button>
          </div>
          {controlMessage && <div className="mt-3 text-xs text-emerald-200 border-l-2 border-emerald-400 pl-2">{controlMessage}</div>}
        </div>
      </div>

      {/* Metrics Banner */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="bg-[#111827] border border-[#1f293d] p-4 rounded-xl shadow-sm">
          <div className="text-xs uppercase font-semibold text-slate-400 flex items-center gap-1.5 mb-1">
            <Camera size={14} className="text-cyan-400" />
            <span>Detector Model</span>
          </div>
          <div className="text-lg font-bold text-cyan-400 font-mono">YOLOv8n Local</div>
        </div>

        <div className="bg-[#111827] border border-[#1f293d] p-4 rounded-xl shadow-sm">
          <div className="text-xs uppercase font-semibold text-slate-400 flex items-center gap-1.5 mb-1">
            <Activity size={14} className="text-emerald-400" />
            <span>Inference Latency</span>
          </div>
          <div className="text-lg font-bold text-white font-mono">{metrics?.latency_ms || 0.0} ms</div>
        </div>

        <div className="bg-[#111827] border border-[#1f293d] p-4 rounded-xl shadow-sm">
          <div className="text-xs uppercase font-semibold text-slate-400 flex items-center gap-1.5 mb-1">
            <Layers size={14} className="text-blue-400" />
            <span>Frame Throughput</span>
          </div>
          <div className="text-lg font-bold text-white font-mono">{metrics?.fps || 0.0} FPS</div>
        </div>

        <div className="bg-[#111827] border border-[#1f293d] p-4 rounded-xl shadow-sm">
          <div className="text-xs uppercase font-semibold text-slate-400 flex items-center gap-1.5 mb-1">
            <Target size={14} className="text-amber-400" />
            <span>Active Tracks</span>
          </div>
          <div className="text-lg font-bold text-white font-mono">{detections.length}</div>
        </div>

        <div className="bg-[#111827] border border-[#1f293d] p-4 rounded-xl shadow-sm">
          <div className="text-xs uppercase font-semibold text-slate-400 flex items-center gap-1.5 mb-1">
            <Eye size={14} className="text-purple-400" />
            <span>Primary Track</span>
          </div>
          <div className="text-lg font-bold text-purple-300 font-mono">
            {detections.length > 0 ? `#${detections[0].tracking_id}` : 'None'}
          </div>
        </div>
      </div>

      {/* Main Grid: Stream on Left, Table on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 bg-[#111827] border border-[#1f293d] rounded-xl overflow-hidden shadow-lg">
          <div className="px-5 py-3 border-b border-[#1f293d] bg-[#0d1424] flex justify-between items-center">
            <span className="text-sm font-bold text-slate-200">Video Perception & ByteTrack HUD</span>
            <span className="text-xs font-mono text-cyan-400">1920×1440 Live Source</span>
          </div>
          <div className="aspect-video bg-black flex items-center justify-center relative">
            <img src="/api/camera/stream" alt="CV Feed" className="w-full h-full object-contain" />
            <div className="absolute bottom-3 left-3 bg-black/70 backdrop-blur px-3 py-1.5 rounded-lg border border-white/10 text-xs font-mono text-slate-300 space-y-0.5">
              <div>Tracker: ByteTrack (Kalman Filter + Hungarian)</div>
              <div>Projection: Pinhole Metric Ground Mapping</div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-6 bg-[#111827] border border-[#1f293d] rounded-xl p-5 shadow-lg flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300">
              Spatial Ground Projection (Pixel → World Metric)
            </h3>
            <span className="text-xs px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              Grid: 0.25m / cell
            </span>
          </div>

          <div className="flex-1 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#0d1424] text-slate-400 uppercase font-mono border-b border-[#1f293d]">
                <tr>
                  <th className="py-2.5 px-3">Track</th>
                  <th className="py-2.5 px-3">BBox (px)</th>
                  <th className="py-2.5 px-3">Conf</th>
                  <th className="py-2.5 px-3">Lateral X</th>
                  <th className="py-2.5 px-3">Distance Y</th>
                  <th className="py-2.5 px-3">Grid Cell</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1f293d]">
                {detections.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-8 text-center text-slate-500">
                      No active person detected in camera view.
                    </td>
                  </tr>
                ) : (
                  detections.map((d, i) => (
                    <tr key={i} className="hover:bg-white/[0.02] font-mono">
                      <td className="py-2.5 px-3 text-cyan-400 font-bold">#{d.tracking_id}</td>
                      <td className="py-2.5 px-3 text-slate-300">[{d.bbox.join(', ')}]</td>
                      <td className="py-2.5 px-3 text-emerald-400 font-bold">{Math.round(d.confidence * 100)}%</td>
                      <td className="py-2.5 px-3 text-slate-200">{d.spatial_x} m</td>
                      <td className="py-2.5 px-3 text-slate-200">{d.spatial_y} m</td>
                      <td className="py-2.5 px-3 text-purple-300 font-bold">
                        ({d.grid_x}, {d.grid_y})
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          <div className="mt-4 pt-3 border-t border-[#1f293d] text-xs text-slate-400 leading-relaxed">
            📐 <strong>Ground Estimation Formula:</strong> Distance Y = (f * H_real) / h_pixel, where calibrated focal length f ≈ 580px and average adult stature H_real ≈ 1.7m. Projected coordinates automatically seed the robotics navigation goal!
          </div>
        </div>
      </div>
    </div>
  );
};
