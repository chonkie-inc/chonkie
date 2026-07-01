import { loader } from "fumadocs-core/source";
import { icons } from "lucide-react";
import { createElement } from "react";
import { buildLoaderSourceInput } from "@/lib/doc-collections";

export const source = loader(buildLoaderSourceInput(), {
  baseUrl: "/",
  icon(icon) {
    if (icon && icon in icons)
      return createElement(icons[icon as keyof typeof icons]);
  },
});
