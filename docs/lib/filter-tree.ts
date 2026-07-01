import type * as PageTree from "fumadocs-core/page-tree";
import type { DocsProductId } from "./docs-products";

const productSlug: Record<DocsProductId, string> = {
  chonkie: "chonkie",
  chonkiejs: "chonkiejs",
};

function folderSlug(folder: PageTree.Folder): string | null {
  const url =
    folder.index?.url ??
    folder.children.find((c) => c.type === "page")?.url;

  if (!url || typeof url !== "string") return null;

  const parts = url.split("/").filter(Boolean);
  return parts[0] ?? null;
}

function findProductFolder(
  nodes: PageTree.Node[],
  slug: string,
): PageTree.Folder | null {
  for (const node of nodes) {
    if (node.type !== "folder") continue;
    if (folderSlug(node) === slug) return node;
    const nested = findProductFolder(node.children, slug);
    if (nested) return nested;
  }
  return null;
}

/** Show only the active product's sidebar pages. */
export function filterPageTreeForProduct(
  tree: PageTree.Root,
  productId: DocsProductId,
): PageTree.Root {
  const slug = productSlug[productId];
  const folder =
    findProductFolder(tree.children, slug) ??
    (tree.fallback ? findProductFolder(tree.fallback.children, slug) : null);

  if (!folder) return tree;

  return {
    ...tree,
    children: folder.children,
  };
}
