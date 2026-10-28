import { useEffect, useState, useCallback } from "react";

import { getLang } from "./i18n/lang";
import { useT } from "./i18n";

export async function api<T>(url: string, init?: { method?: string; body?: unknown }): Promise<T> {
  // Every request carries the UI language so the server can return translated course content.
  url += (url.includes("?") ? "&" : "?") + "lang=" + getLang();
  const res = await fetch(url, {
    method: init?.method ?? (init?.body !== undefined ? "POST" : "GET"),
    headers: init?.body !== undefined ? { "content-type": "application/json" } : undefined,
    body: init?.body !== undefined ? JSON.stringify(init.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok && res.status !== 503) throw new Error((data as any)?.error ?? `HTTP ${res.status}`);
  return data as T;
}

/** Fetch on mount / when `url` changes. `reload` re-fetches without flashing the loading state. */
export function useApi<T>(url: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { lang } = useT(); // re-fetch translated content when the language changes
  const load = useCallback(() => {
    if (!url) return;
    api<T>(url).then((d) => { setData(d); setError(null); }).catch((e) => setError(String(e.message ?? e)));
  }, [url, lang]);
  useEffect(() => { setData(null); load(); }, [load]);
  return { data, error, reload: load };
}
