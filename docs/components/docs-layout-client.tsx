"use client";

import { DocsLayout } from "fumadocs-ui/layouts/docs";
import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";
import type * as PageTree from "fumadocs-core/page-tree";
import { usePathname } from "next/navigation";
import { useMemo, type ReactNode } from "react";
import { ProductSwitcher } from "@/components/product-switcher";
import { filterPageTreeForProduct } from "@/lib/filter-tree";
import { getProductFromPath } from "@/lib/product-routes";

type DocsLayoutClientProps = BaseLayoutProps & {
  tree: PageTree.Root;
  children: ReactNode;
};

export function DocsLayoutClient({
  tree,
  children,
  ...baseOptions
}: DocsLayoutClientProps) {
  const pathname = usePathname() ?? "/docs";
  const productId = getProductFromPath(pathname).id;

  const filteredTree = useMemo(
    () => filterPageTreeForProduct(tree, productId),
    [tree, productId],
  );

  return (
    <DocsLayout
      tree={filteredTree}
      tabs={false}
      containerProps={{
        style: {
          gridTemplateColumns:
            "minmax(0, clamp(1rem, 2vw, 1.5rem)) var(--fd-sidebar-col) minmax(0, 1fr) var(--fd-toc-width) minmax(0, clamp(1rem, 2vw, 1.5rem))",
        },
      }}
      sidebar={{
        banner: <ProductSwitcher key="product-switcher" />,
      }}
      {...baseOptions}
    >
      {children}
    </DocsLayout>
  );
}
