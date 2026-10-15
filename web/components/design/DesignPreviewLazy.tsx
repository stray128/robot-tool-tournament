"use client";

import dynamic from "next/dynamic";

/** Loads the three.js preview only in the browser. */
const DesignPreviewLazy = dynamic(() => import("./DesignPreview"), {
  ssr: false,
  loading: () => <div className="crop-marks aspect-[4/3] bg-paper-2/60" aria-hidden="true" />,
});

export default DesignPreviewLazy;
