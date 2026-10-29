/** Every Nebius job id (`aijob-...`) a text names, once each, in order. */
export function jobIds(md: string): string[] {
  return [...new Set([...md.matchAll(/aijob-[a-z0-9]+/g)].map((m) => m[0]))];
}
