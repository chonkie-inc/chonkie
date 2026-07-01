import { docs } from "collections/server";
import { docs as chonkiejsDocs } from "collections/chonkiejs/server";
import {
  docsProducts,
  getProductById,
  type DocsProduct,
  type DocsProductId,
} from "./docs-products";

/**
 * Registry linking each docs product to its Fumadocs MDX collection.
 *
 * To add a new docs product:
 * 1. Add a `DocsProduct` entry in `lib/docs-products.ts`
 * 2. Add a collection entry below (import the generated `collections/<key>/server` module)
 * 3. For synced external docs: add a workspace in `source.config.ts` and an entry in
 *    `scripts/external-docs.config.mjs`
 */
export interface DocCollectionEntry {
  productId: DocsProductId;
  /** Key passed to `loader({ ... })` */
  loaderKey: string;
  collection: {
    toFumadocsSource: (options?: { baseDir?: string }) => unknown;
  };
  /** Prefix for `toFumadocsSource`, e.g. `chonkiejs` → `/chonkiejs/...` URLs */
  baseDir?: string;
}

const collectionRegistry: DocCollectionEntry[] = [
  {
    productId: "chonkie",
    loaderKey: "docs",
    collection: docs,
  },
  {
    productId: "chonkiejs",
    loaderKey: "chonkiejs",
    collection: chonkiejsDocs,
    baseDir: "chonkiejs",
  },
];

export function buildLoaderSourceInput() {
  return {
    docs: docs.toFumadocsSource(),
    chonkiejs: chonkiejsDocs.toFumadocsSource({ baseDir: "chonkiejs" }),
  } as const;
}

export function getProductForPageUrl(url: string): DocsProduct | undefined {
  return docsProducts.find(
    (product) =>
      url === product.basePath || url.startsWith(`${product.basePath}/`),
  );
}

export function getProductByLoaderKey(
  loaderKey: string,
): DocsProduct | undefined {
  const entry = collectionRegistry.find((item) => item.loaderKey === loaderKey);
  return entry ? getProductById(entry.productId) : undefined;
}
