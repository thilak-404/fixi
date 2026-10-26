export interface SystemStatus {
  state: string;
  camera_active: boolean;
  camera_source?: string;
  camera_fps: number;
  current_person: string;
  current_user_id: string;
  recognition_confidence: number;
  last_interaction: number;
  llm_status: string;
  llm_model: string;
  stt_provider: string;
  tts_provider: string;
  active_people_count: number;
  active_faces_count: number;
  robot_state?: {
    x: number;
    y: number;
    heading: number;
    head_pan: number;
    head_tilt: number;
    is_moving: boolean;
  };
}

export interface WSEvent {
  type: string;
  timestamp: string;
  payload: any;
  is_simulated?: boolean;
}

export interface PointCommandStatus {
  state: string;
  message: string;
  target_id?: string | null;
  target_label?: string | null;
  selection_status: string;
  gesture: {
    class: string;
    confidence: number;
    stable_frames: number;
  };
  candidates: Array<{ object_id: string; label: string; confidence: number; score: number }>;
}

export interface PersonDetection {
  bbox: [number, number, number, number];
  confidence: number;
  tracking_id: number;
  spatial_x: number;
  spatial_y: number;
  grid_x: number;
  grid_y: number;
  latency_ms: number;
}

export interface RoboticsState {
  follow_mode?: boolean;
  robot: {
    grid_col: number;
    grid_row: number;
    world_x: number;
    world_y: number;
    heading_deg: number;
    is_moving: boolean;
  };
  target: {
    grid_col: number;
    grid_row: number;
    label: string;
  } | null;
  person: {
    grid_col: number;
    grid_row: number;
  } | null;
  path: [number, number][];
  path_index: number;
  fsm: {
    state: string;
    duration_s: number;
  };
  grid: {
    width: number;
    height: number;
    resolution: number;
    grid: number[][];
    static_obstacles: [number, number][];
    dynamic_obstacles: [number, number][];
    person_cells: [number, number][];
  };
  planner_metrics?: {
    success: boolean;
    path_length: number;
    cost: number;
    nodes_expanded: number;
    planning_time_ms: number;
  };
}

export interface UserProfile {
  id: string;
  name: string;
  consent_status: string;
  created_at: string;
  last_seen: string;
  has_face_embedding: boolean;
  facts_count?: number;
  episodic_count?: number;
}
