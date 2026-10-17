import Link from "next/link";

/** The Bring Your Part flow's way into the effector tournament (product mode). */
export function EffectorsLink() {
  return (
    <p className="mt-2 text-sm text-ink-3" data-effectors-link>
      Not sure a parallel gripper is the right tool?{" "}
      <Link className="link" href="/effectors/">
        Which effector for this part?
      </Link>{" "}
      compares hands, grippers, suction cups and a soft arm on the training parts (product mode, not part of the
      pre-registered experiment).
    </p>
  );
}
