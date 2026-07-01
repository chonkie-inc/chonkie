import { defineDocs, defineConfig } from "fumadocs-mdx/config";
import * as chonkiejsConfig from "./chonkiejs/source.config";

export const docs = defineDocs({
  dir: "content/docs",
});

export default defineConfig({
  workspaces: {
    chonkiejs: {
      dir: "./chonkiejs",
      config: chonkiejsConfig,
    },
  },
});
