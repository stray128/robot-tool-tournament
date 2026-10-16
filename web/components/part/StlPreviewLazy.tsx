"use client";

import dynamic from "next/dynamic";

/** Loads the three.js STL preview only in the browser. */
const StlPreviewLazy = dynamic(() => import("./StlPreview"), {
  ssr: false,
  loading: () => <div className="crop-marks aspect-[4/3] bg-paper-2/60" aria-hidden="true" />,
});

export default StlPreviewLazy;
