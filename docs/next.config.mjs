import { createMDX } from "fumadocs-mdx/next";

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
      { hostname: "raw.githubusercontent.com" },
    ],
  },
  async redirects() {
    return [
      {
        source: "/python",
        destination: "/chonkie/quick-start",
        permanent: true,
      },
      {
        source: "/python/:path*",
        destination: "/chonkie/:path*",
        permanent: true,
      },
      {
        source: "/docs/overview",
        destination: "/chonkie/quick-start",
        permanent: true,
      },
      {
        source: "/docs/overview/:path*",
        destination: "/chonkie/quick-start",
        permanent: true,
      },
      {
        source: "/overview",
        destination: "/chonkie/quick-start",
        permanent: true,
      },
      {
        source: "/overview/:path*",
        destination: "/chonkie/quick-start",
        permanent: true,
      },
      {
        source: "/docs",
        destination: "/chonkie/quick-start",
        permanent: true,
      },
      {
        source: "/docs/:path*",
        destination: "/:path*",
        permanent: true,
      },
    ];
  },
};

export default withMDX(config);
