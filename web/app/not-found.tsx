import Link from "next/link";

export default function NotFound() {
  return (
    <section className="pt-16">
      <h1 className="display text-[clamp(2.1rem,5.6vw,3.75rem)]">No sheet at this address</h1>
      <p className="prose-measure mt-4 text-lg text-ink-2">
        The page may have moved when the data was re-exported. The benchmark lists every bout and design that exists.
      </p>
      <p className="mt-6">
        <Link href="/benchmark/" className="link">
          Go to the benchmark
        </Link>
      </p>
    </section>
  );
}
