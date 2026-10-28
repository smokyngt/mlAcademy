import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { en } from "./en";
import { fr } from "./fr";
import { setCurrentLang } from "./lang";

export type Lang = "en" | "fr";
export type Key = keyof typeof en;
const DICTS: Record<Lang, Record<Key, string>> = { en, fr };

interface Ctx { lang: Lang; setLang: (l: Lang) => void; t: (key: Key, vars?: Record<string, string | number>) => string }
const I18n = createContext<Ctx>({ lang: "en", setLang: () => {}, t: (k) => en[k] });

function initial(): Lang {
  try {
    const saved = localStorage.getItem("academy.lang");
    if (saved === "fr" || saved === "en") return saved;
  } catch { /* storage may be unavailable */ }
  return navigator.language?.toLowerCase().startsWith("fr") ? "fr" : "en";
}

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initial);
  setCurrentLang(lang); // before children render, so data hooks fetch in the right language
  useEffect(() => { document.documentElement.lang = lang; }, [lang]);
  const setLang = useCallback((l: Lang) => { setLangState(l); try { localStorage.setItem("academy.lang", l); } catch { /* ignore */ } }, []);
  const t = useCallback<Ctx["t"]>((key, vars) => {
    let s = DICTS[lang][key] ?? en[key] ?? key;
    if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, String(v));
    return s;
  }, [lang]);
  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t]);
  return <I18n.Provider value={value}>{children}</I18n.Provider>;
}

export const useT = () => useContext(I18n);

/** Translations for Fumadocs' own UI strings (search box, TOC title, prev/next, theme chooser...). */
export const FUMADOCS_FR = {
  search: "Rechercher", searchNoResult: "Aucun résultat", toc: "Sur cette page", tocNoHeadings: "Aucun titre",
  lastUpdate: "Dernière mise à jour", chooseLanguage: "Choisir la langue", nextPage: "Page suivante",
  previousPage: "Page précédente", chooseTheme: "Thème", editOnGithub: "Modifier sur GitHub",
};
