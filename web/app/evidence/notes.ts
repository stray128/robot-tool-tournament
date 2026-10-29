/**
 * Dated notes on published, pre-registered results. A note never changes a result: the tagged
 * protocol, code and numbers stay as published, and the note says what was found afterwards.
 */

export interface ResultNote {
  /** When the note was added, as shown on the page. */
  date: string;
  /** The note in one sentence, at the head of its Evidence card. */
  lead: string;
  /** The note on the Evidence card. */
  points: { title: string; text: string }[];
  /** The note on a hero's story page, by case id. */
  byHero: Record<string, string>;
}

/** Use-case benchmark v1 (tagged `usecase-v1`), noted 26 Oct 2026. */
export const V1_NOTE: ResultNote = {
  date: "26 Oct 2026",
  lead: "v1 used a top-down wrist; v2 adds wrist tilt and honest tolerances; v1 results stand as published.",
  points: [
    {
      title: "Returns tote",
      text:
        "This result came from a simulated wrist that could only point straight down (no pitch). Under that wrist only soft arms could open the lid, so the soft arm in the recommendation says as much about the wrist as about the task.",
    },
    {
      title: "Pipe coupling",
      text:
        "The insert step counts as done at a depth of at most 4.0 mm, a test equal to the thread start. The published passes sat 6 µm inside it, so a small change to the physics could flip them.",
    },
  ],
  byHero: {
    h1_tote:
      "This result came from a simulated wrist that could only point straight down (no pitch). Under that wrist only soft arms could open the lid, so the soft arm here says as much about the wrist as about the task. The protocol, code and numbers stay as published.",
    h2_coupling:
      "The insert step counts as done at a depth of at most 4.0 mm, a test equal to the thread start. The published passes sat 6 µm inside it, so a small change to the physics could flip them. The protocol, code and numbers stay as published.",
  },
};
