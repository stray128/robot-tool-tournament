"use client";

import { useState } from "react";

/** A shell command or file excerpt with a copy button; `prompt` lines start with "$ " on screen only. */
export function CodeBlock({ code, label, prompt = true }: { code: string; label?: string; prompt?: boolean }) {
  const [copied, setCopied] = useState<"idle" | "done" | "failed">("idle");
  const lines = code.replace(/\n$/, "").split("\n");
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(lines.filter((l) => !l.startsWith("#")).join("\n"));
      setCopied("done");
    } catch {
      setCopied("failed");
    }
    window.setTimeout(() => setCopied("idle"), 1600);
  };
  const button = (
    <button
      type="button"
      onClick={copy}
      className="shrink-0 rounded-md border border-rule-strong bg-surface px-2 py-0.5 text-xs text-ink-2 hover:text-ink"
      aria-label={label ? `Copy: ${label}` : "Copy the commands"}
    >
      {copied === "done" ? "Copied" : copied === "failed" ? "Select and copy" : "Copy"}
    </button>
  );
  return (
    <figure className="min-w-0 overflow-hidden rounded-lg border border-rule bg-paper-2" data-code>
      <figcaption className="flex items-center justify-between gap-3 border-b border-rule px-3 py-1.5 text-xs text-ink-3">
        <span>{label ?? "Commands"}</span>
        {button}
      </figcaption>
      <pre className="overflow-x-auto px-3 py-3 text-[0.8125rem] leading-relaxed">
        <code className="num">
          {lines.map((l, i) => (
            <span key={i} className={`block whitespace-pre ${l.startsWith("#") ? "text-ink-3" : "text-ink"}`}>
              {prompt && l && !l.startsWith("#") && !l.startsWith(" ") ? <span className="select-none text-ink-3">$ </span> : null}
              {l || "\u00a0"}
            </span>
          ))}
        </code>
      </pre>
    </figure>
  );
}
