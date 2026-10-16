/**
 * The site's data contract: the JSON files under public/data/ and the replay headers.
 *
 * Producers: scripts/import-demo.mjs (recorded local demo runs, today) and the
 * repository's export-web step (Nebius runs, later). Every file carries a
 * `schema` name and `version`; lib/validate.ts checks them at build time, so a
 * malformed export fails `pnpm build` instead of shipping a broken page.
 */

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];

// ---------------------------------------------------------------- replays

/** Primitive geom of a replay body, in the body frame (mirrors src/rtt/sim/replay.py). */
export interface ReplayGeom {
  name: string;
  type: "box" | "cylinder" | "capsule" | "plane" | "mesh";
  /** Box half extents (3 values) or plane half size (2 values), metres. */
  half_size?: number[];
  radius?: number;
  /** Cylinder half height along the geom's local Z. */
  half_height?: number;
  /** Capsule half length of the cylindrical part along the geom's local Z. */
  half_length?: number;
  /** Convex mesh piece: flat (x, y, z) vertex triples in the geom frame. */
  vertices?: number[];
  /** Convex mesh piece: flat vertex-index triples, counter-clockwise seen from outside. */
  faces?: number[];
  pos: Vec3;
  quat_xyzw: Quat;
  rgba: [number, number, number, number];
}

export interface ReplayBody {
  name: string;
  geoms: ReplayGeom[];
}

/** `<stem>.json` of a replay: layout of `<stem>.bin` plus the geometry to draw it. */
export interface ReplayHeader {
  format: "rtt-replay";
  version: 1;
  fps: number;
  dtype: "float32";
  byte_order: "little";
  up_axis: "z";
  units: "m";
  bodies: ReplayBody[];
  static: ReplayGeom[];
  n_frames: number;
  n_bodies: number;
  bin: string;
  meta: Record<string, unknown>;
}

// ---------------------------------------------------------------- provenance

export type FailureMode =
  | "missed_grasp"
  | "slip_during_lift"
  | "ejected_on_close"
  | "unstable_hold"
  | "finger_env_collision";

/** Where the physics ran. Only `nebius` runs are scored results. */
export type ComputeProvenance =
  | {
      kind: "local";
      executor: string;
      engine: string;
      host: string;
      wall_s: number | null;
      sim_steps: number | null;
      note: string | null;
    }
  | {
      kind: "nebius";
      job_id: string;
      platform: string;
      preset: string;
      region: string;
      executor: string;
      engine: string;
      wall_s: number;
      cost_usd: number;
      image_digest: string | null;
    };

/** Who or what proposed a finger design. */
export type DesignProvenance =
  | { kind: "manual"; note: string | null }
  | { kind: "random"; sampler: string; seed: number }
  | {
      kind: "llm";
      model: string;
      provider: string;
      tokens_in: number;
      tokens_out: number;
      latency_s: number;
      cost_usd: number;
      round: number | null;
    };

// ---------------------------------------------------------------- bouts.json

export interface PhaseSpan {
  name: "settle" | "descend" | "close" | "lift" | "hold";
  start_s: number;
  /** null for the last phase, which runs until the episode ends. */
  end_s: number | null;
}

/** Task constants every lane of a bout shares (from configs/task_v1.yaml). */
export interface TaskSummary {
  name: string;
  phases: PhaseSpan[];
  success_lift_m: number;
  success_hold_s: number;
  record_tail_s: number;
  max_time_s: number;
  grip_force_n: number;
  max_opening_mm: number;
}

export interface ObjectSummary {
  kind: "box" | "cylinder" | "capsule";
  /** Object-frame extents in mm (box: x y z; cylinder/capsule: diameter, length). */
  size_mm: number[];
}

export interface LaneOutcome {
  success: boolean;
  failure_mode: FailureMode | null;
  max_lift_mm: number;
  hold_s: number;
  flags: string[];
  /** `scorer`: numbers from the simulator's own scorer; `replay-frames`: derived from 30 fps frames. */
  source: "scorer" | "replay-frames";
  wall_s: number | null;
  sim_steps: number | null;
}

export interface Lane {
  id: string;
  /** Replay stem relative to /data, without extension, e.g. "replays/m1/straight". */
  replay: string;
  label: string;
  condition: string;
  entrant: string;
  design_id: string;
  object: ObjectSummary;
  controller_setting: number;
  grasp_height_frac: number | null;
  seed: number;
  n_frames: number;
  duration_s: number;
  outcome: LaneOutcome;
  design_provenance: DesignProvenance;
}

export interface Bout {
  id: string;
  /** Producer tag; an importer replaces only the bouts it produced. */
  source: string;
  kind: "gate" | "training" | "heldout" | "live";
  title: string;
  summary: string;
  featured: boolean;
  recorded_at: string | null;
  task: TaskSummary;
  compute: ComputeProvenance;
  lanes: Lane[];
}

export interface BoutsIndex {
  schema: "rtt-web-bouts";
  version: 1;
  bouts: Bout[];
}

// ---------------------------------------------------------------- designs.json

export interface ParamSpec {
  description: string;
  min?: number;
  max?: number;
  options?: string[];
}

export interface RobustnessCheck {
  name: string;
  condition: string;
  /** Fraction of episodes that succeeded, or null while the check has not run. */
  success_rate: number | null;
  episodes: number;
}

export interface Certificate {
  status: "pending" | "issued";
  issued_at: string | null;
  checks: RobustnessCheck[];
}

export interface DesignFiles {
  stl_left?: string;
  stl_right?: string;
  glb?: string;
  mount_plate?: string;
  certificate_html?: string;
}

export interface Design {
  id: string;
  source: string;
  name: string;
  origin: "manual" | "nemotron" | "random" | "zeroshot";
  rationale: string | null;
  params: Record<string, number | string>;
  valid: boolean;
  errors: string[];
  warnings: string[];
  mass_printed_g: number | null;
  mass_sim_g: number | null;
  bbox_mm: Vec3 | null;
  watertight: boolean | null;
  n_boxes: number | null;
  /** Paths relative to /data. */
  files: DesignFiles;
  provenance: DesignProvenance;
  bouts: string[];
  certificate: Certificate | null;
}

export interface DesignsIndex {
  schema: "rtt-web-designs";
  version: 1;
  params: Record<string, ParamSpec>;
  designs: Design[];
}

// ---------------------------------------------------------------- jobs.json

export interface Job {
  id: string;
  source: string;
  name: string;
  purpose: string;
  kind: "local" | "nebius";
  executor: string;
  platform: string | null;
  preset: string | null;
  region: string | null;
  status: "completed" | "failed" | "cancelled" | "running";
  started_at: string;
  ended_at: string;
  wall_s: number;
  cost_usd: number | null;
  episodes: number | null;
  sim_steps: number | null;
  bouts: string[];
}

export interface JobsIndex {
  schema: "rtt-web-jobs";
  version: 1;
  jobs: Job[];
}

// ---------------------------------------------------------------- results.json

export interface ArmEstimate {
  id: string;
  label: string;
  /** Run-level held-out success rates, 0..1. */
  runs: number[];
  mean: number;
  ci95: [number, number];
}

/** One search run's best-so-far training success after each simulated design. */
export interface S1Run {
  run: string;
  /** Value after design 1..n (n <= budget), 0..1, non-decreasing. */
  best: number[];
  /** Nebius job that ran it, when it ran on Nebius. */
  job_id: string | null;
}

export interface S1Arm {
  id: string;
  label: string;
  runs: S1Run[];
}

/** Where one run's numbers came from (shown under the charts). */
export interface ResultSource {
  arm: string;
  run: string;
  kind: "nebius" | "local";
  job_id: string | null;
  platform: string | null;
  preset: string | null;
  region: string | null;
  executor: string | null;
  wall_s: number | null;
  /** The simulation job's cost (Nebius), not including model calls. */
  cost_usd: number | null;
  /** Model calls the run made (Token Factory); null when not recorded. */
  llm_calls?: number | null;
  /** Their cost, billed separately from the job; null when not recorded. */
  llm_usd?: number | null;
}

/**
 * results.json, version 2. S1 (speed to a good design) is the primary endpoint;
 * H1 (held-out success) is secondary. Version 1 files (`forest` + `curves`) are
 * still read and upgraded by lib/validate.ts.
 */
export interface ResultsIndex {
  schema: "rtt-web-results";
  version: 2;
  /** `synthetic`: layout data, not a result. `pilot`: excluded from confirmatory analysis. */
  status: "synthetic" | "pilot" | "final";
  note: string;
  s1: {
    endpoint: string;
    /** Training-success threshold a run must reach; null while undecided. */
    threshold: number | null;
    /** Where the threshold comes from, e.g. "provisional" or the PREREG tag. */
    threshold_note: string;
    /** Designs per run (B); a run that never reaches the threshold is censored here. */
    budget: number;
    /** Leading designs every arm shares (entrants and seed designs), 0 if none. */
    shared_prefix: number;
    arms: S1Arm[];
  };
  h1: {
    endpoint: string;
    /** `not_run`: the held-out evaluation has not happened; arms are then empty. */
    status: "not_run" | "available";
    arms: ArmEstimate[];
    references: { label: string; value: number }[];
  };
  sources: ResultSource[];
}
