import React, { useEffect, useState } from 'react';
import { User, Brain, Trash2, Plus, ShieldCheck, ShieldAlert } from 'lucide-react';
import { UserProfile } from '../types';

export const MemoryView: React.FC = () => {
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [selectedUserId, setSelectedUserId] = useState<string>('');
  const [memories, setMemories] = useState<any>(null);
  const [newKey, setNewKey] = useState('');
  const [newVal, setNewVal] = useState('');

  const loadUsers = async () => {
    try {
      const res = await fetch('/api/users');
      if (res.ok) {
        const data = await res.json();
        setUsers(data);
        if (data.length > 0 && !selectedUserId) {
          setSelectedUserId(data[0].id);
        }
      }
    } catch (e) {
      console.warn('Failed to load users', e);
    }
  };

  const loadMemories = async (uid: string) => {
    if (!uid) return;
    try {
      const res = await fetch(`/api/memories/${uid}`);
      if (res.ok) {
        const data = await res.json();
        setMemories(data);
      }
    } catch (e) {
      console.warn('Failed to load memories', e);
    }
  };

  useEffect(() => {
    loadUsers();
  }, []);

  useEffect(() => {
    if (selectedUserId) {
      loadMemories(selectedUserId);
    }
  }, [selectedUserId]);

  const handleAddFact = async () => {
    if (!selectedUserId || !newKey.trim() || !newVal.trim()) return;
    try {
      await fetch('/api/memories/facts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: selectedUserId, key: newKey.trim(), value: newVal.trim() }),
      });
      setNewKey('');
      setNewVal('');
      loadMemories(selectedUserId);
    } catch (e) {
      console.warn('Failed to add fact', e);
    }
  };

  const handleDeleteFact = async (key: string) => {
    if (!selectedUserId) return;
    await fetch(`/api/memories/facts/${selectedUserId}/${key}`, { method: 'DELETE' });
    loadMemories(selectedUserId);
  };

  const handleDeleteEpisodic = async (memId: number) => {
    if (!selectedUserId) return;
    await fetch(`/api/memories/episodic/${selectedUserId}/${memId}`, { method: 'DELETE' });
    loadMemories(selectedUserId);
  };

  const handleClearMemories = async () => {
    if (!selectedUserId || !confirm('Permanently wipe all memories for this profile?')) return;
    await fetch(`/api/memories/${selectedUserId}/clear`, { method: 'DELETE' });
    loadMemories(selectedUserId);
  };

  return (
    <div className="space-y-6">
      {/* User Selector Header */}
      <div className="bg-[#111827] border border-[#1f293d] p-5 rounded-xl shadow-lg flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Brain className="text-cyan-400" size={24} />
          <div>
            <h3 className="text-base font-bold text-white">Three-Level Isolated Memory Vault</h3>
            <p className="text-xs text-slate-400">Strict SQLite user_id partition prevents cross-profile leakage.</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold text-slate-300">Active Profile:</label>
          <select
            value={selectedUserId}
            onChange={(e) => setSelectedUserId(e.target.value)}
            className="bg-[#1a2337] border border-[#2a3754] text-slate-100 rounded-lg px-3 py-1.5 text-xs font-semibold focus:outline-none focus:border-cyan-400"
          >
            {users.map((u) => (
              <option key={u.id} value={u.id}>
                {u.name} ({u.id})
              </option>
            ))}
          </select>
          <button
            onClick={handleClearMemories}
            className="px-3 py-1.5 bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 border border-rose-500/30 text-xs font-semibold rounded-lg transition"
          >
            Wipe Memory
          </button>
        </div>
      </div>

      {/* Memory Levels Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Level C: Profile Facts */}
        <div className="bg-[#111827] border border-[#1f293d] p-5 rounded-xl shadow-lg space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
              Level C: Stable Profile Facts (Explicit & Learned)
            </h4>
            <span className="text-xs px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
              {Object.keys(memories?.facts || {}).length} Facts
            </span>
          </div>

          <div className="flex flex-wrap gap-2 min-h-[80px]">
            {Object.entries(memories?.facts || {}).length === 0 ? (
              <div className="text-xs text-slate-500 py-4">No profile facts stored yet.</div>
            ) : (
              Object.entries(memories?.facts || {}).map(([k, v]) => (
                <div
                  key={k}
                  className="bg-[#1a2337] border border-[#2a3854] px-3 py-1.5 rounded-lg text-xs flex items-center gap-2 group"
                >
                  <span className="font-semibold text-cyan-300">{k}:</span>
                  <span className="text-slate-200">{String(v)}</span>
                  <button
                    onClick={() => handleDeleteFact(k)}
                    className="text-slate-500 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))
            )}
          </div>

          {/* Add Fact */}
          <div className="pt-3 border-t border-[#1f293d] flex gap-2">
            <input
              type="text"
              placeholder="Key (e.g. game, age)"
              value={newKey}
              onChange={(e) => setNewKey(e.target.value)}
              className="w-1/3 bg-[#151d30] border border-slate-700/60 rounded-lg px-3 py-1.5 text-xs text-slate-100 placeholder-slate-500"
            />
            <input
              type="text"
              placeholder="Value (e.g. Valorant, 21)"
              value={newVal}
              onChange={(e) => setNewVal(e.target.value)}
              className="flex-1 bg-[#151d30] border border-slate-700/60 rounded-lg px-3 py-1.5 text-xs text-slate-100 placeholder-slate-500"
            />
            <button
              onClick={handleAddFact}
              className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs rounded-lg flex items-center gap-1 transition"
            >
              <Plus size={14} /> Add
            </button>
          </div>
        </div>

        {/* Level B: Episodic Memories */}
        <div className="bg-[#111827] border border-[#1f293d] p-5 rounded-xl shadow-lg space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-xs uppercase font-bold tracking-wider text-cyan-400">
              Level B: Episodic Memories (Past Requests & Interactions)
            </h4>
            <span className="text-xs px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
              {memories?.episodic?.length || 0} Episodes
            </span>
          </div>

          <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
            {(!memories?.episodic || memories.episodic.length === 0) ? (
              <div className="text-xs text-slate-500 py-4">No episodic memories recorded yet.</div>
            ) : (
              memories.episodic.map((ep: any) => (
                <div
                  key={ep.id}
                  className="bg-[#1a2337] border border-[#2a3854] p-3 rounded-lg text-xs flex items-center justify-between group"
                >
                  <div>
                    <div className="font-semibold text-slate-100">{ep.summary}</div>
                    <div className="text-[10px] text-slate-400 mt-0.5">
                      {ep.details || ''} • {new Date(ep.timestamp).toLocaleString()}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteEpisodic(ep.id)}
                    className="text-slate-500 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition p-1"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
