"use client";

import dynamic from "next/dynamic";

/** Loads the three.js design views only in the browser. */
const DesignStudioLazy = dynamic(() => import("./DesignStudio"), {
  ssr: false,
  loading: () => <div className="crop-marks aspect-[4/3] bg-paper-2/60" aria-hidden="true" />,
});

export default DesignStudioLazy;
