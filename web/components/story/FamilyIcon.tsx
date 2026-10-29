/**
 * A small line drawing per tool family, so lists of tools scan by shape. Generic drawings of the
 * mechanism, never of a product.
 */
export function FamilyIcon({ family, className = "size-5" }: { family: string; className?: string }) {
  const f = family.toLowerCase();
  const common = { fill: "none", stroke: "currentColor", strokeWidth: 1.4, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  let body;
  if (f.includes("vacuum") || f.includes("area frame")) {
    // A vacuum area frame: a rigid bar with a row of cups under it.
    body = (
      <>
        <path d="M10 2v4" {...common} />
        <path d="M3 6h14" {...common} />
        <path d="M5 6v3M10 6v3M15 6v3" {...common} />
        <path d="M3 13c0-2 1-3.5 2-3.5s2 1.5 2 3.5zM8 13c0-2 1-3.5 2-3.5s2 1.5 2 3.5zM13 13c0-2 1-3.5 2-3.5s2 1.5 2 3.5z" {...common} />
      </>
    );
  } else if (f.includes("process") || f.includes("nut runner") || f.includes("tamper")) {
    // A process tool: a spindle housing over a socket (nut runner) or a pad (tamper).
    body = (
      <>
        <rect x="6.5" y="2" width="7" height="7" rx="1" {...common} />
        <path d="M10 9v3" {...common} />
        <path d="M7 12h6v4H7z" {...common} />
        <path d="M5 18h10" {...common} />
      </>
    );
  } else if (f.includes("suction")) {
    body = (
      <>
        <path d="M10 2v6" {...common} />
        <path d="M7 8h6l-1 3h-4z" {...common} />
        <path d="M5 15c0-2 2-4 5-4s5 2 5 4z" {...common} />
      </>
    );
  } else if (f.includes("soft")) {
    body = (
      <>
        <path d="M6 2v3" {...common} />
        <path d="M6 5c0 4 2 6 5 7s4 3 3 5-4 1-4-1" {...common} />
      </>
    );
  } else if (f.includes("hand")) {
    body = (
      <>
        <path d="M5 10V6M8 10V3.5M11 10V3.5M14 10V5" {...common} />
        <path d="M5 10c0 4 2 7 5 7s5-3 4-6l2-2" {...common} />
      </>
    );
  } else if (f.includes("designed")) {
    body = (
      <>
        <circle cx="10" cy="6" r="2.5" {...common} />
        <path d="M8 8l-3 5 1 4M12 8l3 5-1 4M10 8.5V17" {...common} />
      </>
    );
  } else if (f.includes("jaw")) {
    body = (
      <>
        <path d="M3 4h14" {...common} />
        <path d="M5 4v12h3V9M15 4v12h-3V9" {...common} />
      </>
    );
  } else {
    // Industrial gripper: a body with two fingers.
    body = (
      <>
        <rect x="4" y="2.5" width="12" height="5" rx="1" {...common} />
        <path d="M6.5 7.5V17M13.5 7.5V17M6.5 13h2M13.5 13h-2" {...common} />
      </>
    );
  }
  return (
    <svg viewBox="0 0 20 20" className={className} aria-hidden="true">
      {body}
    </svg>
  );
}
