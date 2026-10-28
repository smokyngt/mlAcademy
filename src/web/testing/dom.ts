import { JSDOM } from "jsdom";

const KEYS = ["window", "document", "navigator", "HTMLElement", "Node", "Event", "KeyboardEvent", "MouseEvent", "localStorage", "matchMedia"] as const;

/** Installs a jsdom window as globals for interaction tests. `restore` exists for completeness; test files share one instance (bun runs them in one process), so none of them calls it. */
export function installDom() {
  const dom = new JSDOM("<!doctype html><html><body></body></html>", { url: "http://localhost/" });
  const g = globalThis as any;
  const saved = new Map<string, PropertyDescriptor | undefined>();
  for (const k of KEYS) {
    saved.set(k, Object.getOwnPropertyDescriptor(g, k));
    Object.defineProperty(g, k, { value: (dom.window as any)[k], configurable: true, writable: true });
  }
  // jsdom lacks document.oninput; without it React falls back to a focus-based polyfill for text inputs.
  if (!("oninput" in dom.window.document)) (dom.window.document as any).oninput = null;
  // Components read the colour scheme through matchMedia; jsdom does not implement it.
  (dom.window as any).matchMedia ??= (q: string) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, dispatchEvent: () => false, onchange: null });
  g.IS_REACT_ACT_ENVIRONMENT = true;
  return {
    window: dom.window as any,
    restore() {
      for (const [k, d] of saved) d ? Object.defineProperty(g, k, d) : delete g[k];
      delete g.IS_REACT_ACT_ENVIRONMENT;
      dom.window.close();
    },
  };
}

/** React decides at import time whether it is in a browser, so the DOM has to exist before react-dom/client loads: import this module first. */
export const dom = installDom();
