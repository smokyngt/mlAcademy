import { lazy, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Link } from "react-router-dom";
import { VizByName } from "../viz";
import { normalizeMath } from "../lib/math";
import { DynamicCodeBlock } from "fumadocs-ui/components/dynamic-codeblock";
import { useT } from "../i18n";

const InlineNotebook = lazy(() => import("./InlineNotebook"));

const slug = (s: string) => s.toLowerCase().replace(/[^\w]+/g, "-").replace(/^-|-$/g, "");
const text = (n: ReactNode): string => (typeof n === "string" ? n : Array.isArray(n) ? n.map(text).join("") : typeof n === "object" && n && "props" in n ? text((n as any).props.children) : "");

const Lazy = lazy(() => import("./Mermaid"));

/** Renders lesson markdown: GFM, KaTeX, [[lesson-id]] links, ```viz blocks and ```mermaid diagrams. */
export function Markdown({ source, math = true }: { source: string; math?: boolean }) {
  const { t } = useT();
  const body = useMemo(() => normalizeMath(source).replace(/\[\[([\w.-]+)\]\]/g, (_, id: string) => `[${t("md.lesson", { n: Number(id.split(".").pop()) || id })}](/lesson/${id})`), [source, t]);
  return (
    <div className="prose-lesson">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: math }]]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: "ignore" }]]}
        components={{
          h2: ({ children }) => <h2 id={slug(text(children))}>{children}</h2>,
          h3: ({ children }) => <h3 id={slug(text(children))}>{children}</h3>,
          a: ({ href = "", children }) => (href.startsWith("/") ? <Link to={href}>{children}</Link> : <a href={href} target="_blank" rel="noreferrer">{children}</a>),
          pre: ({ children }) => <>{children}</>,
          code: ({ className, children }) => {
            const lang = /language-([\w-]+)/.exec(className ?? "")?.[1];
            const raw = String(children).replace(/\n$/, "");
            if (lang === "viz") return <VizByName name={raw.trim()} />;
            if (lang === "mermaid") return <Suspense fallback={<pre>{raw}</pre>}><Lazy code={raw} /></Suspense>;
            if (lang === "notebook") return <Suspense fallback={<pre>{raw}</pre>}><InlineNotebook source={raw} /></Suspense>;
            if (lang) return <div className="not-prose my-4"><DynamicCodeBlock lang={lang} code={raw} /></div>;
            return raw.includes("\n") ? <pre><code>{raw}</code></pre> : <code>{children}</code>;
          },
        }}
      >
        {body}
      </ReactMarkdown>
    </div>
  );
}

/** Headings (h2) of a markdown source, for the table of contents. */
export function headings(source: string) {
  const out: { id: string; title: string }[] = [];
  let fenced = false;
  for (const line of source.split("\n")) {
    if (line.startsWith("```")) fenced = !fenced;
    const m = !fenced && /^##\s+(.+)$/.exec(line);
    if (m) out.push({ id: slug(m[1].replace(/[`*_$]/g, "")), title: m[1].replace(/[`*_$]/g, "") });
  }
  return out;
}

/** Small inline markdown (math + code) for questions, flashcards and explanations. */
export function Inline({ source }: { source: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: "ignore" }]]} components={{ p: ({ children }) => <span>{children}</span>, code: ({ children }) => <code className="rounded bg-codebg px-1 text-[0.9em]">{children}</code> }}>
      {source}
    </ReactMarkdown>
  );
}
