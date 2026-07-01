export type DocsProductId = "chonkie" | "chonkiejs";

export interface DocsProduct {
  id: DocsProductId;
  label: string;
  description: string;
  basePath: string;
  defaultPage: string;
  badge: string | null;
}

export const docsProducts: DocsProduct[] = [
  {
    id: "chonkie",
    label: "chonkie",
    description: "chonkie on PyPI",
    basePath: "/chonkie",
    defaultPage: "/chonkie/quick-start",
    badge: null,
  },
  {
    id: "chonkiejs",
    label: "chonkiejs",
    description: "@chonkiejs/core on npm",
    basePath: "/chonkiejs",
    defaultPage: "/chonkiejs/getting-started/quick-start",
    badge: null,
  },
];

export function getProductById(id: DocsProductId): DocsProduct {
  const product = docsProducts.find((p) => p.id === id);
  if (!product) throw new Error(`Unknown product: ${id}`);
  return product;
}
