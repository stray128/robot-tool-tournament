"use client";

import dynamic from "next/dynamic";

/** Loads the three.js viewer only in the browser. */
const LibraryViewerLazy = dynamic(() => import("./LibraryViewer"), {
  ssr: false,
  loading: () => (
    <div className="crop-marks grid aspect-[4/3] place-items-center bg-paper-2/60 text-sm text-ink-3">Loading the 3D preview</div>
  ),
});

export default LibraryViewerLazy;
