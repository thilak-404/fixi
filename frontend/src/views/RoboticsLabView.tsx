import React, { useEffect, useRef, useState } from 'react';
import { Play, Pause, Square, RotateCcw, Navigation, ShieldAlert, Cpu, Camera } from 'lucide-react';
import { RoboticsState } from '../types';

export const RoboticsLabView: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [simState, setSimState] = useState<RoboticsState | null>(null);
  const [isAutoStepping, setIsAutoStepping] = useState(false);
  const [feedback, setFeedback] = useState('Simulation idle. Ready for navigation.');

  const handleFollowMe = async () => {
    setFeedback('Camera is locating you and starting simulation follow mode...');
    const res = await fetch('/api/robotics/follow_person', { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      setSimState(data.state);
      setFeedback(data.success ? 'Following detected person. Say "stop" or press Emergency Stop to cancel.' : data.message);
    }
  };

  const fetchState = async () => {
    try {
      const res = await fetch('/api/robotics/state');
      if (res.ok) {
        const data: RoboticsState = await res.json();
        setSimState(data);
      }
    } catch (e) {
      console.warn('Failed to fetch sim state', e);
    }
  };

  useEffect(() => {
    fetchState();
    const interval = setInterval(fetchState, 1000);
    return () => clearInterval(interval);
  }, []);

  // Auto-stepping loop
  useEffect(() => {
    let timer: any = null;
    if (isAutoStepping) {
      timer = setInterval(async () => {
        try {
          const res = await fetch('/api/robotics/step', { method: 'POST' });
          if (res.ok) {
            const data = await res.json();
            setSimState(data.state);
            if (data.result && (data.result.status === 'arrived' || data.result.status === 'completed')) {
              setIsAutoStepping(false);
              setFeedback('Target reached successfully! Visual outcome verified.');
            }
          }
        } catch (e) {
          setIsAutoStepping(false);
        }
      }, 350);
    }
    return () => clearInterval(timer);
  }, [isAutoStepping]);

  // Render 2D Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !simState || !simState.grid) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cols = simState.grid.width || 20;
    const rows = simState.grid.height || 20;
    const cw = canvas.width / cols;
    const ch = canvas.height / rows;

    // Background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Subtle grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
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

    // Grid cells (0=empty, 1=obstacle, 5=inflated)
    const matrix = simState.grid.grid || [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = matrix[r] ? matrix[r][c] : 0;
        if (val === 5) {
          ctx.fillStyle = 'rgba(245, 158, 11, 0.18)';
          ctx.fillRect(c * cw, r * ch, cw, ch);
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.35)';
          ctx.strokeRect(c * cw + 1, r * ch + 1, cw - 2, ch - 2);
        } else if (val === 1) {
          const isDynamic = (simState.grid.dynamic_obstacles || []).some((o) => o[0] === c && o[1] === r);
          ctx.fillStyle = isDynamic ? '#f97316' : '#334155';
          ctx.fillRect(c * cw + 1, r * ch + 1, cw - 2, ch - 2);
          if (isDynamic) {
            ctx.strokeStyle = '#fed7aa';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(c * cw + 2, r * ch + 2, cw - 4, ch - 4);
          }
        }
      }
    }

    // Planned path
    const path = simState.path || [];
    if (path.length > 1) {
      ctx.strokeStyle = '#00f2fe';
      ctx.lineWidth = 3;
      ctx.shadowColor = '#00f2fe';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      path.forEach((pt, i) => {
        const px = pt[0] * cw + cw / 2;
        const py = pt[1] * ch + ch / 2;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Waypoints
      ctx.fillStyle = '#38bdf8';
      path.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt[0] * cw + cw / 2, pt[1] * ch + ch / 2, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Target
    if (simState.target) {
      const tx = simState.target.grid_col * cw + cw / 2;
      const ty = simState.target.grid_row * ch + ch / 2;
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(tx, ty, cw * 0.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('🎯', tx, ty);
    }

    // Person (Vision location)
    if (simState.person && simState.person.grid_col !== null) {
      const px = simState.person.grid_col * cw + cw / 2;
      const py = simState.person.grid_row * ch + ch / 2;
      ctx.fillStyle = '#e11d48';
      ctx.beginPath();
      ctx.arc(px, py, cw * 0.45, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('👤', px, py);
    }

    // Robot with heading indicator
    if (simState.robot) {
      const rx = simState.robot.grid_col * cw + cw / 2;
      const ry = simState.robot.grid_row * ch + ch / 2;

      ctx.shadowColor = '#00f2fe';
      ctx.shadowBlur = 10;
      ctx.fillStyle = '#00f2fe';
      ctx.beginPath();
      ctx.arc(rx, ry, cw * 0.42, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = '#0f172a';
      ctx.beginPath();
      ctx.arc(rx, ry, cw * 0.22, 0, Math.PI * 2);
      ctx.fill();

      // Heading needle
      const rad = (simState.robot.heading_deg * Math.PI) / 180;
      const nx = rx + Math.cos(rad) * (cw * 0.65);
      const ny = ry + Math.sin(rad) * (ch * 0.65);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(rx, ry);
      ctx.lineTo(nx, ny);
      ctx.stroke();
    }
  }, [simState]);

  // Handle cell click to toggle obstacle
  const handleCanvasClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !simState || !simState.grid) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cols = simState.grid.width || 20;
    const rows = simState.grid.height || 20;
    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;

    const clickCol = Math.floor(x / cellW);
    const clickRow = Math.floor(y / cellH);

    const isDynamic = (simState.grid.dynamic_obstacles || []).some((o) => o[0] === clickCol && o[1] === clickRow);
    const action = isDynamic ? 'remove' : 'add';

    try {
      const res = await fetch('/api/robotics/obstacle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ col: clickCol, row: clickRow, action }),
      });
      if (res.ok) {
        const data = await res.json();
        setSimState(data.state);
        setFeedback(`Obstacle ${action === 'add' ? 'added' : 'removed'} at (${clickCol}, ${clickRow}).`);
      }
    } catch (e) {
      console.warn('Failed to toggle obstacle', e);
    }
  };

  const handleNavigateToPerson = async () => {
    setFeedback('Querying vision person coordinates & planning optimal A* path...');
    const res = await fetch('/api/robotics/navigate_person', { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      setSimState(data.state);
      setFeedback(data.success ? 'Path planned! Approach in progress.' : 'Target blocked by obstacles.');
    }
  };

  const handleStep = async () => {
    const res = await fetch('/api/robotics/step', { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      setSimState(data.state);
      if (data.result?.status) setFeedback(`Step: ${data.result.status}`);
    }
  };

  const handleEmergencyStop = async () => {
    setIsAutoStepping(false);
    await fetch('/api/robotics/emergency_stop', { method: 'POST' });
    fetchState();
    setFeedback('🛑 Safety emergency stop engaged! Motion halted.');
  };

  const handleReset = async () => {
    setIsAutoStepping(false);
    await fetch('/api/robotics/reset', { method: 'POST' });
    fetchState();
    setFeedback('Simulation reset to home position (10, 2).');
  };

  const fsmState = simState?.fsm?.state || 'IDLE';

  const fsmNodes = ['IDLE', 'PERCEIVING', 'PLANNING', 'NAVIGATING', 'ARRIVED', 'BLOCKED', 'REPLANNING', 'EMERGENCY_STOP'];

  return (
    <div className="space-y-6">
      <div className="bg-[#111827] border border-cyan-500/30 rounded-xl overflow-hidden shadow-lg">
        <div className="px-4 py-3 border-b border-[#1f293d] bg-[#0d1424] flex items-center justify-between">
          <div className="flex items-center gap-2 text-cyan-300 text-sm font-bold"><Camera size={17} /> Camera person tracking</div>
          <span className="text-xs font-mono text-amber-300">Say: “Fixi, follow me”</span>
        </div>
        <div className="relative aspect-video max-h-80 bg-black flex items-center justify-center">
          <img src="/api/camera/stream" alt="Camera person tracking feed" className="w-full h-full object-contain" />
          <div className="absolute bottom-3 left-3 bg-black/75 border border-cyan-400/30 px-3 py-1.5 text-xs font-mono text-cyan-100">
            {simState?.person ? `Person detected at grid (${simState.person.grid_col}, ${simState.person.grid_row})` : 'Waiting for a person in camera view'}
          </div>
        </div>
      </div>

      {/* FSM Visual State Diagram */}
      <div className="bg-[#111827] border border-[#1f293d] p-4 rounded-xl shadow-lg">
        <div className="flex items-center gap-2 mb-3">
          <Cpu size={16} className="text-cyan-400" />
          <span className="text-xs uppercase font-bold text-slate-300">Robotics Finite State Machine (FSM)</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {fsmNodes.map((st, i) => {
            const isActive = fsmState === st;
            let activeClass = 'bg-[#1e293b] text-slate-400 border-transparent';
            if (isActive) {
              if (st === 'BLOCKED') activeClass = 'bg-rose-500/20 text-rose-300 border-rose-500 shadow-lg shadow-rose-500/30';
              else if (st === 'ARRIVED') activeClass = 'bg-emerald-500/20 text-emerald-300 border-emerald-500 shadow-lg shadow-emerald-500/30';
              else activeClass = 'bg-cyan-500/20 text-cyan-300 border-cyan-400 shadow-lg shadow-cyan-500/30';
            }

            return (
              <React.Fragment key={st}>
                <div className={`px-3 py-1.5 rounded-full text-xs font-bold border transition ${activeClass}`}>
                  {st}
                </div>
                {i < fsmNodes.length - 1 && <span className="text-slate-600 text-xs">➔</span>}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Main Grid: Canvas on Left, Controls & Metrics on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 bg-[#111827] border border-[#1f293d] rounded-xl p-5 shadow-lg flex flex-col items-center">
          <canvas
            ref={canvasRef}
            width={460}
            height={460}
            onClick={handleCanvasClick}
            className="rounded-lg shadow-inner cursor-crosshair border border-slate-800"
          />

          <div className="flex flex-wrap gap-3 mt-4 text-xs text-slate-400">
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded bg-cyan-400"></span> Robot</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded bg-rose-500"></span> Person (Vision)</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded bg-amber-500"></span> Target</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded bg-slate-700"></span> Static Wall</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded bg-orange-500"></span> Dynamic Obstacle</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded bg-amber-500/30 border border-amber-500/50"></span> Safety Inflation</span>
          </div>
          <p className="text-[11px] text-slate-500 mt-2 text-center">
            💡 <em>Click any cell above to toggle a dynamic obstacle and test real-time replanning!</em>
          </p>
        </div>

        {/* Telemetry and Controls (6 cols) */}
        <div className="lg:col-span-6 space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <div className="bg-[#111827] border border-[#1f293d] p-3 rounded-xl">
              <div className="text-[11px] uppercase text-slate-400 font-semibold">Grid Pos</div>
              <div className="text-base font-bold text-cyan-400 font-mono">
                ({simState?.robot.grid_col}, {simState?.robot.grid_row})
              </div>
            </div>
            <div className="bg-[#111827] border border-[#1f293d] p-3 rounded-xl">
              <div className="text-[11px] uppercase text-slate-400 font-semibold">World (x, y)</div>
              <div className="text-base font-bold text-slate-100 font-mono">
                ({simState?.robot.world_x}m, {simState?.robot.world_y}m)
              </div>
            </div>
            <div className="bg-[#111827] border border-[#1f293d] p-3 rounded-xl">
              <div className="text-[11px] uppercase text-slate-400 font-semibold">Heading Angle</div>
              <div className="text-base font-bold text-slate-100 font-mono">{simState?.robot.heading_deg}°</div>
            </div>
            <div className="bg-[#111827] border border-[#1f293d] p-3 rounded-xl">
              <div className="text-[11px] uppercase text-slate-400 font-semibold">Path Length</div>
              <div className="text-base font-bold text-purple-300 font-mono">{simState?.path.length || 0} cells</div>
            </div>
            <div className="bg-[#111827] border border-[#1f293d] p-3 rounded-xl">
              <div className="text-[11px] uppercase text-slate-400 font-semibold">Nodes Expanded</div>
              <div className="text-base font-bold text-emerald-400 font-mono">{simState?.planner_metrics?.nodes_expanded || 0}</div>
            </div>
            <div className="bg-[#111827] border border-[#1f293d] p-3 rounded-xl">
              <div className="text-[11px] uppercase text-slate-400 font-semibold">Planning Latency</div>
              <div className="text-base font-bold text-amber-400 font-mono">{simState?.planner_metrics?.planning_time_ms || 0} ms</div>
            </div>
          </div>

          {/* Action Control Buttons */}
          <div className="bg-[#111827] border border-[#1f293d] p-4 rounded-xl shadow-lg space-y-3">
            <h4 className="text-xs uppercase font-bold text-slate-300 tracking-wider">Simulation Actions</h4>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={handleNavigateToPerson}
                className="px-4 py-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:opacity-90 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-cyan-500/20"
              >
                <Navigation size={14} /> Move Toward Person (Vision Closed-Loop)
              </button>
              <button
                onClick={handleFollowMe}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-emerald-600/20"
              >
                <Navigation size={14} /> Follow Me
              </button>
              <button
                onClick={handleStep}
                className="px-3 py-2 bg-[#1e293b] hover:bg-[#28374f] text-slate-200 rounded-lg text-xs font-semibold flex items-center gap-1.5 border border-slate-700/60"
              >
                <Play size={14} /> Single Step
              </button>
              <button
                onClick={() => setIsAutoStepping(!isAutoStepping)}
                className={`px-3 py-2 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition ${
                  isAutoStepping ? 'bg-amber-600 text-white' : 'bg-[#1e293b] text-slate-200 border border-slate-700/60'
                }`}
              >
                {isAutoStepping ? <Pause size={14} /> : <Play size={14} />} {isAutoStepping ? 'Pause Auto' : 'Auto-Step'}
              </button>
              <button
                onClick={handleEmergencyStop}
                className="px-3 py-2 bg-rose-600 hover:bg-rose-700 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-rose-600/20"
              >
                <ShieldAlert size={14} /> Emergency Stop
              </button>
              <button
                onClick={handleReset}
                className="px-3 py-2 bg-[#1e293b] hover:bg-[#28374f] text-slate-300 rounded-lg text-xs font-semibold flex items-center gap-1.5 border border-slate-700/60"
              >
                <RotateCcw size={14} /> Reset Home
              </button>
            </div>
          </div>

          {/* Feedback Card */}
          <div className="bg-[#0d1424] border border-[#1f293d] p-4 rounded-xl">
            <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-1">
              Closed-Loop Verification Feedback
            </div>
            <div className="text-sm font-semibold text-emerald-400">{feedback}</div>
          </div>
        </div>
      </div>
    </div>
  );
};
