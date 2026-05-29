/** @type {import('next').NextConfig} */
const nextConfig = {
  // Cited page images are plain <img> tags pointing at the public HF resolve URLs,
  // so we don't route them through next/image and need no remotePatterns config.
  reactStrictMode: true,
  // Single-origin proxy: the browser only ever talks to this domain (provenance.icu),
  // so /query and /health are rewritten server-side to the API project — no CORS needed.
  // Local dev is unaffected: there NEXT_PUBLIC_API_URL is unset, so page.tsx calls the
  // API at its absolute localhost URL and never hits these relative paths.
  async rewrites() {
    return [
      { source: "/query", destination: "https://provenance-api-lovat.vercel.app/query" },
      { source: "/health", destination: "https://provenance-api-lovat.vercel.app/health" },
    ];
  },
};

export default nextConfig;
