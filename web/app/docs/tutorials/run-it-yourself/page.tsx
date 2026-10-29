import type { Metadata } from "next";
import Link from "next/link";

import { CodeBlock } from "@/components/docs/CodeBlock";
import { DocSection, DocsShell, Note } from "@/components/docs/DocsShell";
import { formatBytes } from "@/lib/format";
import { GITHUB_URL, repoBlob } from "@/lib/site";

import { featuredHero, getStudioHero } from "../../../studio/data";

export const metadata: Metadata = {
  title: "Run it yourself",
  description: "Install Effector Lab, run the Studio's search on your own plan from the command line or in Docker, and read the report and build pack it writes.",
};

const SECTIONS = [
  { id: "need", title: "What you need" },
  { id: "install", title: "Install" },
  { id: "search", title: "Run the search" },
  { id: "docker", title: "Or run it in Docker" },
  { id: "output", title: "What it writes" },
  { id: "live", title: "Plan live (optional)" },
];

const OPTIONS: [string, string, string][] = [
  ["--out DIR", "required", "Where report.json and the build pack go."],
  ["--workers N", "6", "Simulation processes in parallel."],
  ["--conditions N", "8", "Placements (each with its own seed) every finalist runs the whole task on."],
  ["--screen N", "3", "The first N conditions each tool is screened on, step by step. At most --conditions."],
  ["--finalists N", "4", "Setups that go on to full-task runs."],
  ["--no-library", "off", "Leave out the open library's tools; catalogue and designed tools only."],
  ["--pack", "off", "Also write the build pack of the recommendation (or the closest setup)."],
];

export default function RunItYourselfPage() {
  const featured = featuredHero();
  const page = featured ? getStudioHero(featured.hero) : undefined;
  const timings = page?.provenance.timings ?? null;
  const workers = page?.provenance.workers ?? null;
  const pack = page?.build_pack ?? null;
  const kinds = pack ? [...new Set(pack.files.map((f) => f.kind))] : [];

  return (
    <DocsShell
      current="/docs/tutorials/run-it-yourself/"
      kicker="Tutorial · 20 minutes"
      title="Run it yourself"
      lede={
        <p>
          The worked examples come from one command, and it runs on a laptop. Here is how to run it on your own plan, with Python or Docker, and
          what you get back.
        </p>
      }
      sections={SECTIONS}
    >
      <DocSection id="need" n={1} title="What you need">
        <ul className="grid gap-3 sm:grid-cols-3">
          <li className="panel p-4 text-sm">
            <p className="heading text-base text-ink">A plan</p>
            <p className="mt-1">
              A TaskSpec JSON file from the Studio&apos;s planner (
              <Link className="link" href="/docs/tutorials/plan-a-task/">
                Plan a task
              </Link>
              ).
            </p>
          </li>
          <li className="panel p-4 text-sm">
            <p className="heading text-base text-ink">Python or Docker</p>
            <p className="mt-1">Python 3.12 with uv, or Docker. A CPU is enough: no GPU, no cloud account, no key.</p>
          </li>
          <li className="panel p-4 text-sm">
            <p className="heading text-base text-ink">Some minutes</p>
            <p className="mt-1">
              {timings && typeof timings.total_s === "number" && page
                ? `The ${page.title.toLowerCase()} search took ${Math.round(timings.total_s / 60)} minutes${workers ? ` on ${workers} workers` : ""}: screen, combinations and finalists.`
                : "A search takes minutes on a laptop, depending on the number of steps and tools."}
            </p>
          </li>
        </ul>
      </DocSection>

      <DocSection id="install" n={2} title="Install">
        <p>Clone the repository and install the locked dependencies with uv. The tests run on the CPU and need no network.</p>
        <CodeBlock
          label="Terminal"
          code={`git clone ${GITHUB_URL}.git
cd robot-tool-tournament
make sync
source .venv/bin/activate
# optional: the full test suite
make test`}
        />
      </DocSection>

      <DocSection id="search" n={3} title="Run the search">
        <p>
          Put the plan you downloaded (<code className="num text-ink">taskspec-&lt;id&gt;.json</code>) in the repository folder and run:
        </p>
        <CodeBlock label="Terminal" code={`python -m rtt.studio.search taskspec-<id>.json --out runs/studio/my-task --pack`} />
        <p>
          It screens every tool on every step it could do, combines what passed into single tools, tool changes and two-tool heads, runs the
          finalists through the whole task, ranks them against the best single off-the-shelf tool and writes the result. The options:
        </p>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[34rem] text-left text-sm">
            <thead>
              <tr className="border-b border-rule-strong text-xs text-ink-3">
                <th scope="col" className="py-2 pr-4 font-normal">Option</th>
                <th scope="col" className="py-2 pr-4 font-normal">Default</th>
                <th scope="col" className="py-2 font-normal">What it does</th>
              </tr>
            </thead>
            <tbody>
              {OPTIONS.map(([o, d, t]) => (
                <tr key={o} className="border-b border-rule">
                  <th scope="row" className="num py-2 pr-4 font-normal text-ink">
                    {o}
                  </th>
                  <td className="num py-2 pr-4 text-ink-2">{d}</td>
                  <td className="py-2 text-ink-2">{t}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Note title="Fewer conditions run faster" tone="caution">
          For a first look, <code className="num">--conditions 3 --finalists 2</code> runs in a fraction of the time. Rankings on fewer conditions
          are less certain; the worked examples use the defaults.
        </Note>
      </DocSection>

      <DocSection id="docker" n={4} title="Or run it in Docker">
        <p>The simulation image has the same code and the locked dependencies, so nothing is installed on your machine. Build it once:</p>
        <CodeBlock
          label="Terminal, in the repository folder"
          code={`jobs/nebius/build_image.sh rtt-sim:dev
# a quick check: the first gate passes in a few seconds
docker run --rm rtt-sim:dev m1 --no-files`}
        />
        <p>Then run the search with the current folder mounted at /work:</p>
        <CodeBlock
          label="Terminal"
          code={`docker run --rm -v "$PWD:/work" --entrypoint python rtt-sim:dev \\
  -m rtt.studio.search /work/taskspec-<id>.json --out /work/runs/studio/my-task --pack`}
        />
        <p className="text-sm">The image is about 2 GB. Files it writes belong to the container&apos;s user (root); change their owner afterwards if you need to.</p>
      </DocSection>

      <DocSection id="output" n={5} title="What it writes">
        <ul className="grid gap-3 md:grid-cols-2">
          <li className="panel p-4 text-sm">
            <p className="num text-ink">runs/studio/my-task/report.json</p>
            <p className="mt-1">
              The whole search: the plan, every candidate and why it was ruled out, the success matrix, the combinations, the finalists, the
              recommendation (or the closest setup) and the timings. The command also prints the recommended setup&apos;s id.
            </p>
          </li>
          <li className="panel p-4 text-sm">
            <p className="num text-ink">runs/studio/my-task/build_pack/ and build_pack.zip</p>
            <p className="mt-1">
              With <code className="num">--pack</code>: the files to build the setup, and the same files zipped next to them
              {pack ? ` (the ${page!.title.toLowerCase()} pack is ${formatBytes(pack.bytes)}, ${pack.files.length} files)` : ""}.
            </p>
            {kinds.length ? (
              <p className="mt-2 flex flex-wrap gap-1.5">
                {kinds.map((k) => (
                  <span key={k} className="rounded border border-rule-strong px-1.5 py-px text-xs text-ink-2">
                    {k.replace(/_/g, " ")}
                  </span>
                ))}
              </p>
            ) : null}
          </li>
        </ul>
        <p>
          The story pages on this site are made from the same report by <code className="num text-ink">scripts/export_studio_web.py</code>; see{" "}
          <a className="link" href={repoBlob("docs/STUDIO_SEARCH.md")} rel="noopener">
            Search and ranking
          </a>{" "}
          for every field.
        </p>
      </DocSection>

      <DocSection id="live" n={6} title="Plan live (optional)">
        <p>
          To turn your own words into a plan, run the planner on your machine with your own Nebius Token Factory key. Anything that spends asks
          for a passcode you choose.
        </p>
        <CodeBlock
          label="Terminal 1: the local API"
          code={`cp .env.example .env
# put your key in .env: NEBIUS_API_KEY=...
export RTT_AGENT_PASSCODE_SHA256=$(python -c "import hashlib;print(hashlib.sha256(b'your-passcode').hexdigest())")
rtt serve`}
        />
        <CodeBlock
          label="Terminal 2: the site, pointed at it"
          code={`cd web
pnpm install --frozen-lockfile
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 pnpm build
pnpm start`}
        />
        <p>
          Open <span className="num text-ink">http://127.0.0.1:4173/studio/</span>, describe the task, enter the passcode and press Plan. A plan is
          one model call, or two when the first breaks a rule, on your key;{" "}
          <a className="link" href={repoBlob("docs/STUDIO_ENGINEER.md")} rel="noopener">
            the planner&apos;s guide
          </a>{" "}
          describes the passcode and the spend caps.
        </p>
      </DocSection>
    </DocsShell>
  );
}
