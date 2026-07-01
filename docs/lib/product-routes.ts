import {
  getProductById,
  type DocsProductId,
} from "./docs-products";

/** Maps a logical page key to paths per product (relative to site root). */
const routeMap: Record<string, Partial<Record<DocsProductId, string>>> = {
  "getting-started": {
    chonkie: "chonkie/quick-start",
    chonkiejs: "chonkiejs/getting-started/quick-start",
  },
  installation: {
    chonkie: "chonkie/installation",
    chonkiejs: "chonkiejs/getting-started/installation",
  },
  "token-chunker": {
    chonkie: "chonkie/chunkers/token-chunker",
    chonkiejs: "chonkiejs/chunkers/token-chunker",
  },
  "sentence-chunker": {
    chonkie: "chonkie/chunkers/sentence-chunker",
    chonkiejs: "chonkiejs/chunkers/sentence-chunker",
  },
  "recursive-chunker": {
    chonkie: "chonkie/chunkers/recursive-chunker",
    chonkiejs: "chonkiejs/chunkers/recursive-chunker",
  },
  "semantic-chunker": {
    chonkie: "chonkie/chunkers/semantic-chunker",
    chonkiejs: "chonkiejs/chunkers/semantic-chunker",
  },
  "code-chunker": {
    chonkie: "chonkie/chunkers/code-chunker",
    chonkiejs: "chonkiejs/chunkers/code-chunker",
  },
  "table-chunker": {
    chonkie: "chonkie/chunkers/table-chunker",
    chonkiejs: "chonkiejs/chunkers/table-chunker",
  },
  "fast-chunker": {
    chonkie: "chonkie/chunkers/fast-chunker",
    chonkiejs: "chonkiejs/chunkers/fast-chunker",
  },
  chroma: {
    chonkie: "chonkie/handshakes/chroma-handshake",
    chonkiejs: "chonkiejs/handshakes/chroma",
  },
  changelog: {
    chonkie: "chonkie/changelog",
    chonkiejs: "chonkiejs/changelog",
  },
  troubleshooting: {
    chonkie: "chonkie/troubleshooting",
    chonkiejs: "chonkiejs/troubleshooting",
  },
};

function normalizePath(pathname: string): string {
  const path = pathname.replace(/\/$/, "") || "/";
  if (path.startsWith("/docs")) {
    return path.slice("/docs".length) || "/";
  }
  return path;
}

function getRouteKey(pathSuffix: string): string | null {
  const segments = pathSuffix.replace(/^\//, "").split("/");

  if (segments.includes("quick-start")) return "getting-started";
  if (segments.includes("installation")) return "installation";
  if (segments.includes("changelog")) return "changelog";
  if (segments.includes("troubleshooting")) return "troubleshooting";

  const chunkerIdx = segments.indexOf("chunkers");
  if (chunkerIdx !== -1 && segments[chunkerIdx + 1]) {
    return segments[chunkerIdx + 1].replace(/\.mdx$/, "");
  }

  const handshakeIdx = segments.indexOf("handshakes");
  if (handshakeIdx !== -1 && segments[handshakeIdx + 1]) {
    return segments[handshakeIdx + 1].replace(/-handshake$/, "");
  }

  return null;
}

export function getProductFromPath(pathname: string) {
  const normalized = normalizePath(pathname);

  if (normalized.startsWith("/chonkiejs")) {
    return getProductById("chonkiejs");
  }

  if (normalized.startsWith("/chonkie")) {
    return getProductById("chonkie");
  }

  // Legacy /python URLs still resolve to the chonkie product.
  if (normalized.startsWith("/python")) {
    return getProductById("chonkie");
  }

  return getProductById("chonkie");
}

export function switchProductHref(
  pathname: string,
  targetId: DocsProductId,
): string {
  const current = getProductFromPath(pathname);
  if (current.id === targetId) return pathname;

  const pathSuffix = normalizePath(pathname);
  const routeKey = getRouteKey(pathSuffix);

  if (routeKey && routeMap[routeKey]?.[targetId]) {
    return `/${routeMap[routeKey][targetId]}`;
  }

  return getProductById(targetId).defaultPage;
}
