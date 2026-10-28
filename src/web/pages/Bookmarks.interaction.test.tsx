import { dom } from "../testing/dom"; // must stay first: installs jsdom before react-dom/client loads
import { afterEach, beforeAll, beforeEach, describe, expect, test } from "bun:test";
import { act } from "react";
import type { Root } from "react-dom/client";
import { buttonByText, click, flush, mockServer } from "../testing/helpers";

const { createRoot } = await import("react-dom/client");
const { MemoryRouter } = await import("react-router-dom");
const { LangProvider } = await import("../i18n");
const { BookmarksPage } = await import("./Bookmarks");

let root: Root;
let host: HTMLElement;
let server: ReturnType<typeof mockServer>;
let rows: { lessonId: string; title: string; note: string; createdAt: number }[];

const mount = () => act(() => { root.render(<MemoryRouter><LangProvider><BookmarksPage /></LangProvider></MemoryRouter>); });

beforeAll(() => { dom.window.localStorage.setItem("academy.lang", "en"); });
beforeEach(() => {
  host = document.createElement("div"); document.body.appendChild(host); root = createRoot(host);
  rows = [{ lessonId: "f2n.06", title: "Shapes", note: "revisit", createdAt: Date.UTC(2026, 9, 1) }, { lessonId: "f2n.07", title: "Gradient descent", note: "", createdAt: Date.UTC(2026, 9, 2) }];
  server = mockServer((r) => {
    if (r.method === "DELETE") { rows = rows.filter((b) => !r.url.includes(b.lessonId)); return { json: { ok: true } }; }
    return { json: rows };
  });
});
afterEach(() => { act(() => root.unmount()); host.remove(); server.restore(); });

describe("bookmarks page", () => {
  test("lists bookmarks with links to their lessons and shows notes only when present", async () => {
    mount(); await flush();
    const links = [...host.querySelectorAll("a")];
    expect(links.map((a) => a.getAttribute("href"))).toEqual(["/lesson/f2n.06", "/lesson/f2n.07"]);
    expect(host.textContent).toContain("revisit");
    expect(host.querySelectorAll("li")).toHaveLength(2);
  });

  test("removing a bookmark calls DELETE for that lesson and refreshes the list", async () => {
    mount(); await flush();
    click(host.querySelectorAll("li button")[0]);
    await flush();
    const del = server.requests.find((r) => r.method === "DELETE")!;
    expect(del.url).toStartWith("/api/bookmarks/f2n.06");
    expect(host.querySelectorAll("li")).toHaveLength(1);
    expect(host.querySelector("a")!.getAttribute("href")).toBe("/lesson/f2n.07");
  });

  test("removing the last bookmark shows the empty state", async () => {
    rows = [rows[0]];
    mount(); await flush();
    click(host.querySelector("li button")!);
    await flush();
    expect(host.querySelectorAll("li")).toHaveLength(0);
    expect(host.querySelector("p.text-muted")).not.toBeNull();
  });

  test("the page is rendered in the stored language", async () => {
    dom.window.localStorage.setItem("academy.lang", "fr");
    try {
      mount(); await flush();
      expect(host.querySelector("h1")!.textContent).not.toBe("Bookmarks");
      expect(buttonByText(host, /retirer|supprimer|remove/i)).toBeDefined();
    } finally { dom.window.localStorage.setItem("academy.lang", "en"); }
  });
});
