"use client";

import dynamic from "next/dynamic";

/** Loads the hero replay (three.js) only in the browser. */
const HeroReplayLazy = dynamic(() => import("./HeroReplay"), {
  ssr: false,
  loading: () => <div className="aspect-video bg-paper-2/60" aria-hidden="true" />,
});

export default HeroReplayLazy;
