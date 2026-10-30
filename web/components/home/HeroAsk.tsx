"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useId, useState } from "react";

import { announceLeave } from "@/components/replay/leaving";
import { PREFILL_KEY } from "@/lib/site";


export interface ExampleChip {
  href: string;
  title: string;
  setting: string;
}

/**
 * "What should your robot do?": the text goes to the Studio planner, prefilled. It travels in
 * this tab's session storage, never in the URL, so it never reaches a server log; when storage is
 * unavailable the page says so and keeps the text here. This static site calls no model: the
 * planner replays the recorded examples, takes a plan built by hand, and points to running the
 * planner on your own machine for new words. The sentence beside the button comes from the page
 * (liveCopy in lib/live_copy.ts), which says whether a build with the hosted runner also points to
 * its live panel; it is a prop so this client code stays free of the runner client.
 */
export function HeroAsk({ examples, note }: { examples: ExampleChip[]; note: string }) {
  const router = useRouter();
  const [text, setText] = useState("");
  const [hint, setHint] = useState<string | null>(null);
  const id = useId();

  const submit = () => {
    const clean = text.trim();
    if (!clean) {
      setHint("Describe the job first, or open one of the examples below.");
      return;
    }
    try {
      sessionStorage.setItem(PREFILL_KEY, clean);
    } catch {
      setHint("This browser does not let the page carry your words to the planner. Copy them, then open the planner in the Studio.");
      return;
    }
    // The hero replay stops first, so its frames do not compete with the navigation.
    announceLeave();
    router.push("/studio/#plan");
  };

  return (
    <div className="mt-8" data-hero-ask>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
        className="panel flex flex-col gap-2 p-2 focus-within:border-accent/70"
      >
        <label htmlFor={id} className="sr-only">
          What should your robot do?
        </label>
        <textarea
          id={id}
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setHint(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={3}
          maxLength={2000}
          placeholder="What should your robot do? For example: open the hinged lid of a tote on the bench, put a carton and a flat pack in it, then close the lid."
          className="w-full resize-none bg-transparent px-3 py-2 text-ink placeholder:text-ink-3 focus:outline-none"
        />
        <div className="flex flex-wrap items-center justify-between gap-2 border-t border-rule px-2 pt-2">
          <span className="text-xs text-ink-3">
            {note}{" "}
            <Link href="/docs/tutorials/run-it-yourself/" className="text-ink-2 underline underline-offset-2 hover:text-ink">
              run it yourself
            </Link>
            .
          </span>
          <button type="submit" className="btn btn-primary shrink-0">
            Plan it in the Studio
          </button>
        </div>
      </form>
      <p className="mt-2 min-h-5 text-sm text-pending" role="status">
        {hint}
      </p>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-sm">
        <span className="text-ink-3">Or see a worked example:</span>
        {examples.map((ex) => (
          <Link
            key={ex.href}
            href={ex.href}
            className="inline-flex items-center gap-1.5 rounded-full border border-rule-strong px-3 py-1 text-ink no-underline hover:border-accent hover:text-accent"
            data-example-chip
          >
            {ex.title}
            <span className="text-xs text-ink-3">{ex.setting}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
