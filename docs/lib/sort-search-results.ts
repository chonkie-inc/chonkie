import type { SortedResult } from "fumadocs-core/search";
import { docsProducts } from "@/lib/docs-products";

type SearchResultLike = Pick<SortedResult, "url" | "breadcrumbs">;

function getProductRank(item: SearchResultLike): number {
  for (let i = 0; i < docsProducts.length; i++) {
    const product = docsProducts[i];
    if (
      item.url?.startsWith(`${product.basePath}/`) ||
      item.url === product.basePath ||
      item.breadcrumbs?.[0] === product.label
    ) {
      return i;
    }
  }

  return docsProducts.length;
}

/** Order search hits: chonkie, then chonkiejs, then anything else. */
export function sortSearchResults<T extends SearchResultLike>(items: T[]): T[] {
  return items
    .map((item, index) => ({ item, index, rank: getProductRank(item) }))
    .sort((a, b) => a.rank - b.rank || a.index - b.index)
    .map(({ item }) => item);
}
