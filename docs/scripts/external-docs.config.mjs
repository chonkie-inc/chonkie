/**
 * External docs synced at build/dev time into workspace folders.
 *
 * To add another synced docs package:
 * 1. Add an entry here (envVar + outDir)
 * 2. Register a workspace in `source.config.ts`
 * 3. Import `collections/<outDir>/server` in `lib/doc-collections.ts`
 * 4. Add a product in `lib/docs-products.ts`
 */
export const externalDocSources = [
  {
    id: "chonkiejs",
    envVar: "CHONKIEJS_DOCS_URL",
    outDir: "chonkiejs",
    transforms: [
      {
        file: "content/docs/changelog.mdx",
        replacements: [
          [
            "<GithubReleases />",
            '<GithubReleases src="/data/releases-js.json" />',
          ],
        ],
      },
    ],
  },
];
