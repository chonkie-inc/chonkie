import type { ReactNode } from "react";
import { DocsLayoutClient } from "@/components/docs-layout-client";
import { source } from "@/lib/source";
import { baseOptions } from "@/lib/layout.shared";

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <DocsLayoutClient tree={source.pageTree} {...baseOptions}>
      {children}
    </DocsLayoutClient>
  );
}
