import React, { useState, useEffect } from 'react';
import { Camera, Bot, Brain, Activity, BarChart3, Radio, ShieldAlert, RotateCcw } from 'lucide-react';
import { SystemStatus, WSEvent } from './types';
import { DashboardView } from './views/DashboardView';
import { CVLabView } from './views/CVLabView';
import { RoboticsLabView } from './views/RoboticsLabView';
import { MemoryView } from './views/MemoryView';
import { BenchmarksView } from './views/BenchmarksView';

type Tab = 'dashboard' | 'cv' | 'robotics' | 'memory' | 'benchmarks';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('dashboard');
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [events, setEvents] = useState<WSEvent[]>([]);
  const [robotState, setRobotState] = useState<string>('IDLE');

  // Status polling
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch('/api/status');
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
        }
      } catch (e) {
        console.warn('Status poll error', e);
      }
    };
    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const pollRobot = async () => {
      try {
        const res = await fetch('/api/robotics/state');
        if (res.ok) {
          const data = await res.json();
          setRobotState(data?.fsm?.state || 'IDLE');
        }
      } catch (e) {
        setRobotState('DISCONNECTED');
      }
    };
    pollRobot();
    const interval = setInterval(pollRobot, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        void fetch('/api/robotics/emergency_stop', { method: 'POST' });
      }
      if (event.key.toLowerCase() === 'r' && !['INPUT', 'TEXTAREA'].includes((event.target as HTMLElement)?.tagName)) {
        void fetch('/api/robotics/reset', { method: 'POST' });
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  // WebSocket event bus connection
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/events`;
    let ws: WebSocket | null = null;

    const connect = () => {
      try {
        ws = new WebSocket(wsUrl);
        ws.onmessage = (ev) => {
          try {
            const data: WSEvent = JSON.parse(ev.data);
            setEvents((prev) => [data, ...prev.slice(0, 49)]);
          } catch (err) {}
        };
        ws.onclose = () => {
          setTimeout(connect, 3000);
        };
      } catch (e) {
        setTimeout(connect, 3000);
      }
    };

    connect();
    return () => {
      if (ws) ws.close();
    };
  }, []);

  const handleTriggerAction = async (action: string) => {
    if (action === 'wave') {
      await fetch('/api/robot/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'wave' }),
      });
    } else if (action === 'move_toward_me') {
      await fetch('/api/robotics/navigate_person', { method: 'POST' });
    }
  };

  const handleEmergencyStop = async () => {
    await fetch('/api/robotics/emergency_stop', { method: 'POST' });
    setRobotState('EMERGENCY_STOP');
  };

  const handleReset = async () => {
    await fetch('/api/robotics/reset', { method: 'POST' });
    setRobotState('IDLE');
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#070b14] text-slate-100 hud-shell">
      {/* Top Navigation Bar */}
      <header className="border-b border-[#1f293d] bg-[#0d1424] sticky top-0 z-50 px-6 py-3">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="px-3 py-1 rounded-lg bg-cyan-400/10 border border-cyan-300/70 font-black text-sm tracking-widest text-cyan-200 shadow-md shadow-cyan-500/20">
              FIXI
            </div>
            <div>
              <h1 className="text-sm font-bold text-white tracking-wide">
                Integrated Computer Vision & Robotics Platform
              </h1>
              <p className="text-[11px] text-slate-400">100% Local Inference • Zero Cloud Dependencies</p>
            </div>
          </div>

          {/* Quick HUD Indicators */}
          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-1.5 bg-[#111827] px-2.5 py-1 rounded-lg border border-[#1f293d]">
              <span className={`w-2 h-2 rounded-full ${status?.camera_active ? 'bg-emerald-400' : 'bg-rose-500'}`}></span>
              <span className="text-slate-300">
                Cam: {status?.camera_active ? (status.camera_source === 'synthetic' ? 'Synthetic' : 'Active') : 'Offline'}
              </span>
            </div>
            <div className="flex items-center gap-1.5 bg-[#111827] px-2.5 py-1 rounded-lg border border-[#1f293d]">
              <Radio size={12} className="text-cyan-400" />
              <span className="text-slate-300">LLM: {status?.llm_model || 'qwen2.5'}</span>
            </div>
            <div className="flex items-center gap-1.5 bg-[#111827] px-2.5 py-1 rounded-lg border border-[#1f293d]">
              <span className="text-cyan-400 font-bold">Privacy: Local Only</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="hud-badge hud-badge-green"><span className="hud-dot" /> LOCAL ONLY</span>
            <span className="hud-badge hud-badge-amber">SIMULATION</span>
            <button
              onClick={handleReset}
              className="hud-icon-button"
              title="Reset simulation (R)"
              aria-label="Reset simulation"
            >
              <RotateCcw size={15} />
            </button>
            <button
              onClick={handleEmergencyStop}
              className="hud-stop-button"
              title="Emergency stop (Escape)"
              aria-label="Emergency stop"
            >
              <ShieldAlert size={15} /> STOP
            </button>
          </div>
        </div>
      </header>

      <div className="hud-status-line" role="status">
        <span className="text-cyan-300">SYSTEM</span>
        <span>{robotState === 'IDLE' ? 'Ready for a stable pointing gesture.' : `Robotics state: ${robotState}.`}</span>
        <span className="hud-key-hint">ESC stop · R reset</span>
      </div>

      {/* Lab Tabs Navigation */}
      <nav className="border-b border-[#1f293d] bg-[#0e1629] px-6">
        <div className="max-w-7xl mx-auto flex space-x-1">
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`px-4 py-3 text-xs font-bold transition flex items-center gap-2 border-b-2 ${
              activeTab === 'dashboard'
                ? 'border-cyan-400 text-cyan-400 bg-white/[0.03]'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Bot size={16} /> Live Companion
          </button>

          <button
            onClick={() => setActiveTab('cv')}
            className={`px-4 py-3 text-xs font-bold transition flex items-center gap-2 border-b-2 ${
              activeTab === 'cv'
                ? 'border-cyan-400 text-cyan-400 bg-white/[0.03]'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Camera size={16} /> Computer Vision Lab
          </button>

          <button
            onClick={() => setActiveTab('robotics')}
            className={`px-4 py-3 text-xs font-bold transition flex items-center gap-2 border-b-2 ${
              activeTab === 'robotics'
                ? 'border-cyan-400 text-cyan-400 bg-white/[0.03]'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Activity size={16} /> Robotics Lab (2D Canvas)
          </button>

          <button
            onClick={() => setActiveTab('memory')}
            className={`px-4 py-3 text-xs font-bold transition flex items-center gap-2 border-b-2 ${
              activeTab === 'memory'
                ? 'border-cyan-400 text-cyan-400 bg-white/[0.03]'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Brain size={16} /> Memory Vault
          </button>

          <button
            onClick={() => setActiveTab('benchmarks')}
            className={`px-4 py-3 text-xs font-bold transition flex items-center gap-2 border-b-2 ${
              activeTab === 'benchmarks'
                ? 'border-cyan-400 text-cyan-400 bg-white/[0.03]'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <BarChart3 size={16} /> Academic Benchmarks
          </button>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6">
        {activeTab === 'dashboard' && (
          <DashboardView status={status} events={events} onTriggerAction={handleTriggerAction} />
        )}
        {activeTab === 'cv' && <CVLabView />}
        {activeTab === 'robotics' && <RoboticsLabView />}
        {activeTab === 'memory' && <MemoryView />}
        {activeTab === 'benchmarks' && <BenchmarksView />}
      </main>
    </div>
  );
};
