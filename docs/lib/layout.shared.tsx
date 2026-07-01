import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";

export const baseOptions: BaseLayoutProps = {
  nav: {
    title: (
      <>
        <img
          src="https://www.chonkie.ai/chonkies/chonkie_icon.svg"
          alt="Chonkie"
          width={28}
          height={28}
          className="rounded-sm"
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
