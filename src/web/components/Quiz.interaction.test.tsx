import { dom } from "../testing/dom"; // must stay first: installs jsdom before react-dom/client loads
import { afterEach, beforeAll, beforeEach, describe, expect, test } from "bun:test";
import { act } from "react";
import type { Root } from "react-dom/client";
import type { AnswerResult, QuestionPublic } from "@shared/types";

// Loaded after jsdom is installed: react-dom/client and react-router-dom decide whether a DOM exists when they are first evaluated.
const { createRoot } = await import("react-dom/client");
const { MemoryRouter } = await import("react-router-dom");
const { LangProvider } = await import("../i18n");
const { Question } = await import("./Quiz");

const realFetch = globalThis.fetch;
let root: Root;
let host: HTMLElement;
let requests: { url: string; body: any }[] = [];

/** Replaces fetch with a canned server; `reply` decides the status and JSON body for each call. */
function mockServer(reply: (body: any) => { status?: number; json: unknown }) {
  (globalThis as any).fetch = async (url: string, init?: { body?: string }) => {
    const body = init?.body ? JSON.parse(init.body) : undefined;
    requests.push({ url, body });
    const r = reply(body);
    return { ok: (r.status ?? 200) < 400, status: r.status ?? 200, json: async () => r.json } as Response;
  };
}

const mcq = { id: "q1", kind: "mcq", prompt: "2 + 2 = ?", options: ["3", "4", "5"] } as unknown as QuestionPublic;
const numeric = { id: "q2", kind: "numeric", prompt: "Square root of 9?", options: [] } as unknown as QuestionPublic;
const result = (over: Partial<AnswerResult> = {}): AnswerResult => ({ correct: true, explanation: "Because arithmetic.", answer: 1, mastery: 0.6, recommendation: null, ...over });

function mount(q: QuestionPublic, onAnswered?: (r: AnswerResult) => void) {
  act(() => { root.render(<MemoryRouter><LangProvider><Question q={q} index={0} onAnswered={onAnswered} /></LangProvider></MemoryRouter>); });
}
const buttons = () => [...host.querySelectorAll("button")] as HTMLButtonElement[];
const check = () => buttons().find((b) => b.className.includes("btn-primary"))!;
const flush = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });
const click = (el: Element) => act(() => { el.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true })); });
const type = (input: HTMLInputElement, value: string) => act(() => {
  Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, "value")!.set!.call(input, value);
  input.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
});
const radios = () => [...host.querySelectorAll("input[type=radio]")] as HTMLInputElement[];

beforeAll(() => { dom.window.localStorage.setItem("academy.lang", "en"); });
beforeEach(() => { host = document.createElement("div"); document.body.appendChild(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); (globalThis as any).fetch = realFetch; requests = []; });

describe("multiple choice question", () => {
  test("Check stays disabled until an option is picked", () => {
    mount(mcq);
    expect(check().disabled).toBe(true);
    click(radios()[1]);
    expect(check().disabled).toBe(false);
  });

  test("submits the chosen index, reports it, and locks the options", async () => {
    mockServer(() => ({ json: result({ correct: true, answer: 1 }) }));
    let seen: AnswerResult | undefined;
    mount(mcq, (r) => { seen = r; });
    click(radios()[1]);
    click(check());
    await flush();
    expect(requests).toHaveLength(1);
    expect(requests[0].url).toStartWith("/api/questions/q1/answer");
    expect(requests[0].body).toEqual({ response: 1 });
    expect(host.querySelector("[role=status]")!.textContent).toContain("Because arithmetic.");
    expect(seen?.correct).toBe(true);
    expect(radios().every((i) => i.disabled)).toBe(true);
  });

  test("a wrong answer shows the review recommendation with a link to the lesson", async () => {
    const recommendation = { conceptId: "c", conceptTitle: "Addition", reason: "You missed 2 recent questions.", review: [{ conceptId: "p", title: "Counting", lessonId: "f2n.01", score: 0.25 }, { conceptId: "z", title: "Never seen", lessonId: null, score: null }] };
    mockServer(() => ({ json: result({ correct: false, answer: 1, recommendation: recommendation as any }) }));
    mount(mcq);
    click(radios()[0]);
    click(check());
    await flush();
    const link = host.querySelector("a")!;
    expect(link.getAttribute("href")).toBe("/lesson/f2n.01");
    expect(link.textContent).toBe("Counting");
    expect(host.textContent).toContain("Never seen");
    expect(host.textContent).toContain("25");
  });

  test("Retry clears the result and the selection", async () => {
    mockServer(() => ({ json: result({ correct: false }) }));
    mount(mcq);
    click(radios()[2]);
    click(check());
    await flush();
    click(buttons().find((b) => !b.className.includes("btn-primary"))!);
    expect(host.querySelector("[role=status]")).toBeNull();
    expect(radios()[2].checked).toBe(false);
    expect(check().disabled).toBe(true);
  });

  test("a server error is shown and the question can be answered again", async () => {
    mockServer(() => ({ status: 500, json: { error: "database is locked" } }));
    mount(mcq);
    click(radios()[0]);
    click(check());
    await flush();
    expect(host.textContent).toContain("database is locked");
    expect(host.querySelector("[role=status]")).toBeNull();
    expect(check().disabled).toBe(false);
  });
});

describe("numeric question", () => {
  test("Check is disabled when empty and non-numeric text never reaches the server", async () => {
    mockServer(() => ({ json: result() }));
    mount(numeric);
    const input = host.querySelector("input.input") as HTMLInputElement;
    expect(check().disabled).toBe(true);
    type(input, "abc");
    click(check());
    await flush();
    expect(requests).toHaveLength(0);
  });

  test("Enter submits the typed number", async () => {
    mockServer(() => ({ json: result({ answer: 3 }) }));
    mount(numeric);
    const input = host.querySelector("input.input") as HTMLInputElement;
    type(input, "3.0");
    act(() => { input.dispatchEvent(new dom.window.KeyboardEvent("keydown", { key: "Enter", bubbles: true })); });
    await flush();
    expect(requests[0].body).toEqual({ response: 3 });
    expect(input.disabled).toBe(true);
  });

  test("a wrong numeric answer reveals the expected value", async () => {
    mockServer(() => ({ json: result({ correct: false, answer: 3 }) }));
    mount(numeric);
    type(host.querySelector("input.input") as HTMLInputElement, "4");
    click(check());
    await flush();
    expect(host.querySelector("[role=status]")!.textContent).toContain("3");
  });
});
