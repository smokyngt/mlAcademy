import { expect, test } from "bun:test";
import { renderToString } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { DocsLayout } from "fumadocs-ui/layouts/docs";
import { DocsBody, DocsPage, DocsTitle } from "fumadocs-ui/layouts/docs/page";
import { FumadocsProvider } from "./FumadocsProvider";
import { buildDocsSource } from "./source";
import { LangProvider } from "../i18n";
import { App } from "../App";

// Regression: the previous provider needed a data router and crashed the whole app with a blank page.
const catalog = [{ id: "c", title: "Course", summary: "", stage: "I", kind: "authored" as const, part: "math" as const, lessons: [{ id: "f2n.01", title: "Functions", minutes: 40, position: 0, status: "new" as const }] }];

test("Fumadocs layout renders under a declarative BrowserRouter-style router", () => {
  const html = renderToString(
    <MemoryRouter initialEntries={["/lesson/f2n.01"]}>
      <LangProvider>
        <FumadocsProvider>
          <DocsLayout tree={buildDocsSource(catalog).pageTree} nav={{ title: "ML Academy", url: "/" }}>
            <DocsPage toc={[{ title: "Intro", url: "#intro", depth: 2 }]}><DocsTitle>Functions</DocsTitle><DocsBody>body</DocsBody></DocsPage>
          </DocsLayout>
        </FumadocsProvider>
      </LangProvider>
    </MemoryRouter>,
  );
  expect(html).toContain("Functions");
  expect(html).toContain("/lesson/f2n.01");
});

test("the whole app renders its first screen without a data router", () => {
  const html = renderToString(<MemoryRouter initialEntries={["/"]}><LangProvider><App /></LangProvider></MemoryRouter>);
  expect(html).toContain("ML Academy");
});
