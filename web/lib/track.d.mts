export declare const POSE_STRIDE: 7;
export declare function referenceFrame(fps: number, descendStartS: number, nFrames: number): number;
export declare function liftTrack(
  frames: Float32Array,
  nBodies: number,
  bodyIndex: number,
  refFrame: number,
): Float32Array;
export declare function holdTrack(lift: Float32Array, fps: number, threshold: number): Float32Array;
export declare function summarizeTracks(
  lift: Float32Array,
  hold: Float32Array,
): { maxLift: number; maxHold: number };
