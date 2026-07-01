"use client";

import { useEffect, useState, useRef } from "react";
import { createHighlighter, type Highlighter } from "shiki";

interface Release {
  id: number;
  name: string;
  tag: string;
  published_at: string;
  body: string;
  html_url: string;
}

export function GithubReleases({ src = "/data/releases.json" }: { src?: string }) {
  const [releases, setReleases] = useState<Release[]>([]);
  const [loading, setLoading] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(src)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load releases");
        return res.json();
      })
      .then((data) => {
        setReleases(data);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!containerRef.current || releases.length === 0) return;
    highlightAllCodeBlocks(containerRef.current);
  }, [releases]);

  if (loading) {
    return <div className="text-fd-muted-foreground py-4">Loading releases...</div>;
  }

  if (releases.length === 0) {
    return <div className="text-fd-muted-foreground py-4">No releases found.</div>;
  }

  return (
    <div className="space-y-6 not-prose" ref={containerRef}>
      {releases.map((release) => (
        <div
          key={release.id}
          className="rounded-xl border border-fd-border bg-fd-card p-5"
        >
          <div className="flex items-center justify-between mb-3">
            <a
              href={release.html_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-lg font-semibold text-fd-foreground hover:text-fd-primary transition-colors"
            >
              {release.name}
            </a>
            <span className="text-xs text-fd-muted-foreground shrink-0 ml-3">
              {new Date(release.published_at).toLocaleDateString("en-US", {
                month: "long",
                day: "numeric",
                year: "numeric",
              })}
            </span>
          </div>
          <div
            className="release-body text-sm text-fd-muted-foreground leading-relaxed"
            dangerouslySetInnerHTML={{ __html: formatBody(release.body) }}
          />
        </div>
      ))}
    </div>
  );
}

let highlighterPromise: Promise<Highlighter> | null = null;

function getHighlighter() {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({
      themes: ["github-dark"],
      langs: ["python", "bash", "javascript", "typescript", "json", "yaml", "text"],
    });
  }
  return highlighterPromise;
}

async function highlightAllCodeBlocks(container: HTMLElement) {
  const wrappers = container.querySelectorAll<HTMLElement>(".release-code-wrapper");
  if (wrappers.length === 0) return;

  const highlighter = await getHighlighter();

  for (const wrapper of wrappers) {
    const preEl = wrapper.querySelector<HTMLElement>("pre.release-code");
    if (!preEl) continue;

    const lang = preEl.dataset.lang || "text";
    const code = preEl.textContent || "";
    if (!code.trim()) continue;

    const supportedLangs = highlighter.getLoadedLanguages();
    const resolvedLang = supportedLangs.includes(lang) ? lang : "text";

    try {
      const html = highlighter.codeToHtml(code, {
        lang: resolvedLang,
        theme: "github-dark",
      });
      preEl.outerHTML = html.replace("<pre", '<pre class="release-code"');
    } catch {
      // leave as plain text
    }
  }
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function processInline(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong class="text-fd-foreground font-medium">$1</strong>')
    .replace(/`([^`]+)`/g, '<code class="release-inline-code">$1</code>')
    .replace(/@([\w-]+)(?:\[bot\])?/g, '<a href="https://github.com/$1" target="_blank" rel="noopener noreferrer" class="text-fd-primary hover:underline">@$1</a>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-fd-primary hover:underline">$1</a>')
    .replace(/(?<!")(https:\/\/github\.com\/[\w-]+\/[\w-]+\/pull\/\d+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer" class="text-fd-primary hover:underline">#$1</a>')
    .replace(/#(https:\/\/github\.com\/[\w-]+\/[\w-]+\/pull\/)(\d+)<\/a>/g, '#$2</a>');
}

function formatBody(body: string): string {
  let text = body.replace(/\r\n/g, "\n");

  // Remove "Full Changelog" line
  text = text.replace(/\*\*Full Changelog\*\*:.*$/gm, "");

  // Remove top-level h1 headings (redundant with release name)
  text = text.replace(/^# .+$/gm, "");

  // Strip all raw HTML tables (complex nested ones from GitHub)
  text = text.replace(/<table[\s\S]*?<\/table>/gi, "");

  // Convert HTML images to proper img tags with styling
  text = text.replace(
    /<img[^>]*?src="([^"]+)"[^>]*?\/?>/gi,
    '<img src="$1" class="release-img" loading="lazy" />'
  );

  // Remove <p> wrappers but keep inner content
  text = text.replace(/<p[^>]*>([\s\S]*?)<\/p>/gi, "$1");

  // Remove other stray HTML tags (br, div, etc.)
  text = text.replace(/<br\s*\/?>/gi, "\n");
  text = text.replace(/<\/?div[^>]*>/gi, "");

  // Convert markdown images: ![alt](url)
  text = text.replace(
    /!\[([^\]]*)\]\((https?:\/\/[^)]+)\)/g,
    '<img src="$2" alt="$1" class="release-img" loading="lazy" />'
  );

  // Convert badge images that are inside links: [![alt](img_url)](link_url)
  text = text.replace(
    /\[<img[^>]*?>\]\((https?:\/\/[^)]+)\)/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer" class="text-fd-primary hover:underline">link</a>'
  );

  // Convert GitHub admonitions: > [!TIP], > [!CAUTION], etc.
  text = text.replace(
    /^> \[!(TIP|CAUTION|NOTE|WARNING|IMPORTANT)\]\n((?:^>.*\n?)*)/gm,
    (_, type, content) => {
      const inner = content.replace(/^> ?/gm, "").trim();
      const colors: Record<string, string> = {
        TIP: "border-green-500/50 bg-green-500/5",
        NOTE: "border-blue-500/50 bg-blue-500/5",
        CAUTION: "border-red-500/50 bg-red-500/5",
        WARNING: "border-yellow-500/50 bg-yellow-500/5",
        IMPORTANT: "border-purple-500/50 bg-purple-500/5",
      };
      const color = colors[type] || "border-fd-border bg-fd-muted";
      const processedInner = processInline(inner).replace(/\n/g, "<br/>");
      return `<div class="border-l-4 ${color} rounded-r-lg px-4 py-3 my-3"><span class="font-medium text-fd-foreground text-xs uppercase tracking-wide">${type}</span><div class="mt-1">${processedInner}</div></div>\n`;
    }
  );

  // Convert code blocks (triple backticks) — extract and replace with placeholders
  const codeBlocks: string[] = [];
  text = text.replace(
    /```(\w*)\n([\s\S]*?)```/g,
    (_, lang, code) => {
      const idx = codeBlocks.length;
      const cleaned = code.trim().replace(/\n{3,}/g, "\n\n");
      const escaped = escapeHtml(cleaned);
      codeBlocks.push(`<div class="release-code-wrapper"><pre class="release-code" data-lang="${lang}"><code>${escaped}</code></pre><button class="release-copy-btn" onclick="navigator.clipboard.writeText(this.parentElement.querySelector('code').textContent).then(()=>{this.textContent='Copied!';setTimeout(()=>{this.textContent='Copy'},1500)})">Copy</button></div>`);
      return `\n%%CODEBLOCK_${idx}%%\n`;
    }
  );

  // Convert markdown tables (lines starting with |)
  text = text.replace(
    /((?:^\|.+\|\s*\n)+)/gm,
    (tableBlock) => {
      const rows = tableBlock.trim().split("\n");
      if (rows.length < 2) return tableBlock;

      let html = '<div class="overflow-x-auto my-3"><table class="release-table">';
      rows.forEach((row, i) => {
        if (row.match(/^\|\s*[-:]+[-| :]*$/)) return;
        const cells = row
          .split("|")
          .slice(1, -1)
          .map((c) => c.trim());
        const tag = i === 0 ? "th" : "td";
        html +=
          "<tr>" +
          cells.map((c) => `<${tag}>${processInline(c)}</${tag}>`).join("") +
          "</tr>";
      });
      html += "</table></div>\n";
      return html;
    }
  );

  // Convert ## and ### headings
  text = text.replace(/^### (.+)$/gm, '<h4 class="release-h4">$1</h4>');
  text = text.replace(/^## (.+)$/gm, '<h3 class="release-h3">$1</h3>');

  // Convert horizontal rules
  text = text.replace(/^---$/gm, '<hr class="my-4 border-fd-border" />');

  // Convert list items and process inline content within them
  text = text.replace(/^\* (.+)$/gm, (_, content) => `<li>${processInline(content)}</li>`);
  text = text.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul class="release-list">$1</ul>');

  // Process remaining inline markdown on plain text lines
  const lines = text.split("\n");
  const processed = lines.map((line) => {
    if (line.startsWith("<") || line.trim() === "" || line.match(/^%%CODEBLOCK_\d+%%$/)) {
      return line;
    }
    return processInline(line);
  });
  text = processed.join("\n");

  // Convert remaining newlines
  text = text.replace(/\n{3,}/g, "<br/><br/>");
  text = text.replace(/\n{2}/g, "<br/>");
  text = text.replace(/\n/g, " ");

  // Reinsert code blocks (preserved from newline collapsing)
  text = text.replace(/%%CODEBLOCK_(\d+)%%/g, (_, idx) => codeBlocks[parseInt(idx)]);

  return text.trim();
}
