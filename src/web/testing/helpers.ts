import { act } from "react";
import { dom } from "./dom";

export type Req = { url: string; method: string; body: any };

/** Replaces fetch with a canned server. `reply` sees each request and returns the status and JSON body; every call is recorded. */
export function mockServer(reply: (req: Req) => { status?: number; json: unknown }) {
  const requests: Req[] = [];
  const real = globalThis.fetch;
  (globalThis as any).fetch = async (url: string, init?: { method?: string; body?: string }) => {
    const req = { url, method: init?.method ?? "GET", body: init?.body ? JSON.parse(init.body) : undefined };
    requests.push(req);
    const r = reply(req);
    return { ok: (r.status ?? 200) < 400, status: r.status ?? 200, json: async () => r.json } as Response;
  };
  return { requests, restore: () => { (globalThis as any).fetch = real; } };
}

export const flush = () => act(async () => { await new Promise((r) => setTimeout(r, 0)); });
export const click = (el: Element) => act(() => { el.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true })); });
export const buttonByText = (host: HTMLElement, re: RegExp) => [...host.querySelectorAll("button")].find((b) => re.test(b.textContent ?? "")) as HTMLButtonElement;
