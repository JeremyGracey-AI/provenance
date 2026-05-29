/** @type {import('next').NextConfig} */
const nextConfig = {
  // Cited page images are plain <img> tags pointing at the public HF resolve URLs,
  // so we don't route them through next/image and need no remotePatterns config.
  reactStrictMode: true,
};

export default nextConfig;
