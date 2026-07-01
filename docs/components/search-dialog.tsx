"use client";

import {
  SearchDialog,
  SearchDialogClose,
  SearchDialogContent,
  SearchDialogFooter,
  SearchDialogHeader,
  SearchDialogIcon,
  SearchDialogInput,
  SearchDialogList,
  SearchDialogOverlay,
} from "fumadocs-ui/components/dialog/search";
import type { DefaultSearchDialogProps } from "fumadocs-ui/components/dialog/search-default";
import { useI18n } from "fumadocs-ui/contexts/i18n";
import { useDocsSearch } from "fumadocs-core/search/client";
import { fetchClient } from "fumadocs-core/search/client/fetch";
import { use, useMemo } from "react";
import { sortSearchResults } from "@/lib/sort-search-results";

let staticImport:
  | Promise<typeof import("fumadocs-core/search/client/orama-static")>
  | undefined;

export function ChonkieSearchDialog({
  type = "static",
  api,
  delayMs,
  links = [],
  footer,
  ...props
}: DefaultSearchDialogProps) {
  const { locale } = useI18n();
  let client;
  if (type === "static") {
    client = use(
      staticImport ??= import("fumadocs-core/search/client/orama-static"),
    ).oramaStaticClient({ from: api, locale });
  } else {
    client = fetchClient({ api, locale });
  }

  const { search, setSearch, query } = useDocsSearch({ client, delayMs });

  const items = useMemo(() => {
    if (query.data === "empty") {
      if (links.length === 0) return null;
      return links.map(([name, link]) => ({
        type: "page" as const,
        id: name,
        content: name,
        url: link,
      }));
    }

    if (!query.data) return query.data;
    return sortSearchResults(query.data);
  }, [query.data, links]);

  return (
    <SearchDialog
      search={search}
      onSearchChange={setSearch}
      isLoading={query.isLoading}
      {...props}
    >
      <SearchDialogOverlay />
      <SearchDialogContent>
        <SearchDialogHeader>
          <SearchDialogIcon />
          <SearchDialogInput />
          <SearchDialogClose />
        </SearchDialogHeader>
        <SearchDialogList items={items ?? null} />
      </SearchDialogContent>
      {footer ? <SearchDialogFooter>{footer}</SearchDialogFooter> : null}
    </SearchDialog>
  );
}
