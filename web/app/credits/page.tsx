import { existsSync, readFileSync } from "node:fs";
import path from "node:path";

import type { Metadata } from "next";
import type { ReactNode } from "react";

import { EffectorCredits } from "@/components/effectors/EffectorCredits";
import { PageHeader } from "@/components/PageHeader";
import { TitleBlock } from "@/components/TitleBlock";
import { creditGroups, parseAssetManifest, splitCredit, type AssetManifest } from "@/lib/assets";
import { APP_NAME } from "@/lib/site";
import type { AssetRole } from "@/lib/types";

export const metadata: Metadata = {
  title: "Credits",
  description: "The open models, scans, software and fonts this site and its simulations are built on, with their licences.",
};

/** The served asset manifest, read at build time (an absent file means no assets are served yet). */
function servedAssets(): AssetManifest {
  const file = path.join(process.cwd(), "public", "assets", "manifest.json");
  if (!existsSync(file)) return { format: "rtt-asset-manifest", version: 1, assets: {} };
  return parseAssetManifest(JSON.parse(readFileSync(file, "utf8")), "public/assets/manifest.json");
}

const ROLE_LABELS: Record<AssetRole, string> = {
  finger_left: "left jaw",
  finger_right: "right jaw",
  gripper_frame: "gripper frame",
  carriage: "jaw carriage",
  object: "part",
  prop: "prop",
};

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section className="mt-14 grid gap-x-10 gap-y-3 md:grid-cols-[14rem_minmax(0,1fr)]" aria-labelledby={id}>
      <h2 id={id} className="heading text-xl">
        {title}
      </h2>
      <div className="prose-measure grid min-w-0 gap-4 text-ink-2">{children}</div>
    </section>
  );
}

/**
 * Licence texts for the licences of shipped third-party material. CC BY 4.0 (section
 * 3(a)(1)(C)) requires the credit to carry the licence's URI or a link to it.
 */
const LICENCE_URLS: Record<string, string> = {
  "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
  "CERN-OHL-P-2.0": "https://ohwr.org/cern_ohl_p_v2.txt",
};

function Credit({ name, licence, children }: { name: ReactNode; licence: string; children?: ReactNode }) {
  const href = LICENCE_URLS[licence];
  return (
    <div className="grid gap-x-4 gap-y-1 border-b border-rule pb-3 sm:grid-cols-[minmax(0,1fr)_9rem]">
      <div className="min-w-0">
        <p className="text-ink">{name}</p>
        {children ? <div className="mt-1 grid gap-1 text-sm">{children}</div> : null}
      </div>
      <p className="num text-sm text-ink sm:text-right">
        {href ? (
          <a className="link" href={href} rel="noreferrer license">
            {licence}
          </a>
        ) : (
          licence
        )}
      </p>
    </div>
  );
}

function Ext({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a className="link break-words" href={href} rel="noreferrer">
      {children}
    </a>
  );
}

const SOFTWARE: { name: string; href: string; licence: string; use: string }[] = [
  {
    name: "MuJoCo (Google DeepMind)",
    href: "https://github.com/google-deepmind/mujoco",
    licence: "Apache-2.0",
    use: "Physics engine for every simulated grasp",
  },
  {
    name: "three.js",
    href: "https://github.com/mrdoob/three.js",
    licence: "MIT",
    use: "3D views; its RoomEnvironment lights the studio look, with no downloaded HDRI",
  },
  {
    name: "meshoptimizer (Arseny Kapoulkine)",
    href: "https://github.com/zeux/meshoptimizer",
    licence: "MIT",
    use: "Mesh compression of the served models, decoded by the decoder bundled with three.js",
  },
  {
    name: "glTF Transform (Don McCurdy)",
    href: "https://github.com/donmccurdy/glTF-Transform",
    licence: "MIT",
    use: "Build-time GLB compression",
  },
  {
    name: "React Three Fiber and drei (Poimandres)",
    href: "https://github.com/pmndrs/react-three-fiber",
    licence: "MIT",
    use: "React bindings for the 3D views",
  },
  {
    name: "Next.js and React",
    href: "https://github.com/vercel/next.js",
    licence: "MIT",
    use: "Static site",
  },
  {
    name: "zustand",
    href: "https://github.com/pmndrs/zustand",
    licence: "MIT",
    use: "Shared replay clock",
  },
  {
    name: "Tailwind CSS",
    href: "https://github.com/tailwindlabs/tailwindcss",
    licence: "MIT",
    use: "Styles",
  },
  {
    name: "CoACD",
    href: "https://github.com/SarahWeiii/CoACD",
    licence: "MIT",
    use: "Convex decomposition of scanned objects",
  },
  {
    name: "trimesh",
    href: "https://github.com/mikedh/trimesh",
    licence: "MIT",
    use: "Mesh processing",
  },
  {
    name: "Pillow",
    href: "https://github.com/python-pillow/Pillow",
    licence: "MIT-CMU",
    use: "Textures of the served models",
  },
  {
    name: "imageio",
    href: "https://github.com/imageio/imageio",
    licence: "BSD-2-Clause",
    use: "Video frames of the rendered replays",
  },
  {
    name: "manifold3d",
    href: "https://github.com/elalish/manifold",
    licence: "Apache-2.0",
    use: "Solid modelling of printable fingers",
  },
];

export default function CreditsPage() {
  const groups = creditGroups(servedAssets().assets);

  return (
    <>
      <PageHeader
        title="Credits"
        lede={
          <p>
            {APP_NAME} is built on open models, scans and software. This page lists each one, its licence and what was
            changed. Everything else, including every designed finger and every industrial part, is our own work under
            Apache-2.0.
          </p>
        }
      />

      <Section id="models" title="3D models in replays">
        {groups.length > 0 ? (
          groups.map((g) => {
            const { text, anchor } = splitCredit(g.attribution);
            return (
              <Credit
                key={`${g.attribution}-${g.license}`}
                name={
                  anchor ? (
                    <>
                      {text}{" "}
                      <a href={`#${anchor}`} className="link whitespace-nowrap">
                        (full credit)
                      </a>
                    </>
                  ) : (
                    text
                  )
                }
                licence={g.license}
              >
                <p>
                  Used for: {g.roles.map((r) => ROLE_LABELS[r]).join(", ")} ({g.count} {g.count === 1 ? "file" : "files"}).
                </p>
                <p>Modified: {g.changes.join(", ")}.</p>
              </Credit>
            );
          })
        ) : (
          <p>No third-party models are served yet. The replays on this site draw only geometry generated by this project.</p>
        )}
      </Section>

      <Section id="gripper" title="Gripper">
        <Credit
          name={
            <>
              Robonine, SO-ARM 102, <Ext href="https://github.com/roboninecom/SO-ARM-102">https://github.com/roboninecom/SO-ARM-102</Ext>,
              CC BY 4.0
            </>
          }
          licence="CC-BY-4.0"
        >
          <p>
            Copyright (c) 2026 Robonine. Licensed under{" "}
            <Ext href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</Ext>, provided as is, without warranties.
          </p>
          <p>
            Its simulation meshes (<span className="num">simulation/urdf/meshes</span>, commit{" "}
            <span className="num">190eb162</span>) are the source of the gripper frame and jaw carriages drawn in replays, and
            of the stock finger entrant. In replays and videos it is called &ldquo;an open-source parallel gripper&rdquo;.
          </p>
          <p>
            Modified: cut at the jaw root plane into carriage and blade; re-posed into the simulator&apos;s palm and jaw frames
            (the right clamp moved 1.0 mm so the jaws are symmetric); split into convex collision pieces for the stock finger
            (CoACD threshold 0.02, seed 0, at most 40 pieces); re-encoded as GLB, and compressed for the web.
          </p>
        </Credit>
        <Credit name="SO-ARM 102 fabrication CAD (models/)" licence="CERN-OHL-P-2.0">
          <p>
            Not used or shipped: the print pack&apos;s jaw interface is our own model, drawn from documented dimensions. A
            print file derived from this CAD would carry the CERN-OHL-P-2.0 notice and a list of the changes.
          </p>
        </Credit>
      </Section>

      <Section id="effectors" title="Effector models (product mode)">
        <EffectorCredits />
      </Section>

      <Section id="gso" title="Scanned objects">
        <Credit
          name={
            <>
              Google Scanned Objects by Google Research (Downs et al., ICRA 2022,{" "}
              <Ext href="https://arxiv.org/abs/2204.11918">arXiv:2204.11918</Ext>), CC BY 4.0
            </>
          }
          licence="CC-BY-4.0"
        >
          <p>
            Licensed under <Ext href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</Ext>, provided as is, without
            warranties.
          </p>
          <p>
            Used for held-out test objects and props, through the MuJoCo models by Kevin Zakka (
            <Ext href="https://github.com/kevinzakka/mujoco_scanned_objects">kevinzakka/mujoco_scanned_objects</Ext>, commit{" "}
            <span className="num">6ff8d275</span>; scans CC BY 4.0, MJCF XML MIT).
          </p>
          <p>
            Modified: for simulation, split into convex pieces (CoACD threshold 0.02, seed 0) and re-posed into the object
            frame; for display, meshes simplified to 50%, textures re-encoded as WebP and the result re-encoded as GLB.
          </p>
          <p>Scans whose textures show a brand mark are drawn untextured.</p>
        </Credit>
      </Section>

      <Section id="models-ai" title="Design models">
        <Credit name="NVIDIA Nemotron 3 models, served by Nebius Token Factory" licence="Model card">
          <p>
            Designs, hypotheses and agent turns are generated through an API. No weights are included; each model&apos;s
            licence and terms are on its model card.
          </p>
        </Credit>
      </Section>

      <Section id="software" title="Software">
        {SOFTWARE.map((s) => (
          <Credit key={s.name} name={<Ext href={s.href}>{s.name}</Ext>} licence={s.licence}>
            <p>{s.use}</p>
          </Credit>
        ))}
      </Section>

      <Section id="fonts" title="Fonts">
        <Credit name="Archivo (Omnibus-Type)" licence="OFL-1.1" />
        <Credit name="B612 Mono (Airbus, Intactile Design)" licence="OFL-1.1" />
      </Section>

      <TitleBlock title="Credits" drawing="/credits" sheet="Appendix" />
    </>
  );
}
