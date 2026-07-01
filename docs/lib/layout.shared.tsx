import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";

export const baseOptions: BaseLayoutProps = {
  nav: {
    title: (
      <>
        <img
          src="/assets/logo/chonkie_logo_br_transparent_bg.png"
          alt="Chonkie"
          width={28}
          height={28}
          className="rounded-sm h-7 w-auto object-contain"
        />
        <span>Chonkie</span>
      </>
    ),
  },
  githubUrl: "https://github.com/chonkie-inc/chonkie",
  links: [
    {
      text: "Discord",
      url: "https://discord.gg/Q6zkP8w6ur",
    },
  ],
};
