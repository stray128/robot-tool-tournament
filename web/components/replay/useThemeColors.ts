"use client";

import { useEffect, useState } from "react";

/** Colours the 3D views take from the page's CSS tokens, so they follow the theme. */
export interface SceneColors {
  ink: string;
  ink2: string;
  ink3: string;
  accent: string;
  stock: string;
  paper2: string;
  paper3: string;
  pass: string;
  fail: string;
}

const FALLBACK: SceneColors = {
  ink: "#15181d",
  ink2: "#49515e",
  ink3: "#6f7784",
  accent: "#2340d8",
  stock: "#c4c9be",
  paper2: "#e7e9e3",
  paper3: "#d9ddd4",
  pass: "#17784a",
  fail: "#bb3127",
};

function read(): SceneColors {
  if (typeof window === "undefined") return FALLBACK;
  const css = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  return {
    ink: v("--ink", FALLBACK.ink),
    ink2: v("--ink-2", FALLBACK.ink2),
    ink3: v("--ink-3", FALLBACK.ink3),
    accent: v("--accent", FALLBACK.accent),
    stock: v("--stock", FALLBACK.stock),
    paper2: v("--paper-2", FALLBACK.paper2),
    paper3: v("--paper-3", FALLBACK.paper3),
    pass: v("--pass", FALLBACK.pass),
    fail: v("--fail", FALLBACK.fail),
  };
}

/** Current theme colours; updates when the theme attribute or the OS colour scheme changes. */
export function useThemeColors(): SceneColors {
  const [colors, setColors] = useState<SceneColors>(FALLBACK);
  useEffect(() => {
    const update = () => setColors(read());
    update();
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", update);
    return () => {
      observer.disconnect();
      mq.removeEventListener("change", update);
    };
  }, []);
  return colors;
}
