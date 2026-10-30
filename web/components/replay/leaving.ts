/**
 * A replay's clock stops when the visitor leaves the page through the site's own navigation. On a
 * slow machine (software WebGL, a busy laptop) every replay frame can take most of the main thread,
 * and the router's navigation, which React renders at a low priority, only gets what is left: a
 * click on "Plan it in the Studio" or a header link could then take seconds to land, or not land.
 */

type Listener = () => void;

const listeners = new Set<Listener>();

/** Says that an in-app navigation is starting (call it right before router.push). */
export function announceLeave(): void {
  for (const listener of [...listeners]) listener();
}

const stripSlash = (p: string) => (p.length > 1 ? p.replace(/\/+$/, "") : p);

/**
 * True when this click follows a link to another page of this site in this tab: the primary
 * button with no modifier key (those open a new tab or window), no other target, no download, the
 * same origin, and another path (a hash on this page or this page again keeps the page).
 */
export function leavesPage(e: Pick<MouseEvent, "button" | "metaKey" | "ctrlKey" | "shiftKey" | "altKey" | "target">, here: Pick<Location, "href">): boolean {
  if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return false;
  const target = e.target as { closest?: (selector: string) => unknown } | null;
  const a = (typeof target?.closest === "function" ? target.closest("a[href]") : null) as HTMLAnchorElement | null;
  if (!a) return false;
  if ((a.target && a.target !== "_self") || a.hasAttribute("download")) return false;
  let to: URL;
  let from: URL;
  try {
    from = new URL(here.href);
    to = new URL(a.getAttribute("href") ?? "", from);
  } catch {
    return false;
  }
  return to.origin === from.origin && stripSlash(to.pathname) !== stripSlash(from.pathname);
}

/**
 * Calls `onLeave` when an in-app navigation starts: a click (mouse, touch, pen or keyboard) on a
 * link to another page of the site, or announceLeave(). The click listener runs in the capture
 * phase, before the link's own handler starts the navigation. Returns the unsubscribe function.
 */
export function onLeave(listener: Listener): () => void {
  listeners.add(listener);
  const click = (e: MouseEvent) => {
    if (leavesPage(e, window.location)) listener();
  };
  document.addEventListener("click", click, true);
  return () => {
    listeners.delete(listener);
    document.removeEventListener("click", click, true);
  };
}
