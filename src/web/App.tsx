import { useEffect, useRef, useState } from "react";
import { NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { FumadocsProvider } from "./docs/FumadocsProvider";
import { DocsShell } from "./docs/DocsShell";
import { FUMADOCS_FR, useT, type Key } from "./i18n";
import { Dashboard } from "./pages/Dashboard";
import { Catalog } from "./pages/Catalog";
import { CoursePage } from "./pages/Course";
import { LessonPage } from "./pages/Lesson";
import { ConceptsPage } from "./pages/Concepts";
import { ReviewPage } from "./pages/Review";
import { LabPage } from "./pages/Lab";
import { SearchPage } from "./pages/Search";
import { BookmarksPage } from "./pages/Bookmarks";
import { AssessmentPage } from "./pages/Assessment";
import { CapstonePage } from "./pages/Capstone";
import { RoadmapPage } from "./pages/Roadmap";
import { SourcesPage } from "./pages/Sources";
import { ProfilePage } from "./pages/Profile";
import { ProjectsPage, ProjectPage } from "./pages/Projects";

const nav: readonly (readonly [string, Key])[] = [
  ["/", "nav.dashboard"],
  ["/profile", "nav.profile"],
  ["/courses", "nav.courses"],
  ["/concepts", "nav.concepts"],
  ["/review", "nav.review"],
  ["/lab", "nav.lab"],
  ["/projects", "nav.projects"],
  ["/roadmap", "nav.roadmap"],
  ["/sources", "nav.sources"],
  ["/capstone", "nav.capstone"],
  ["/bookmarks", "nav.bookmarks"],
];

export function App() {
  const { t, lang, setLang } = useT();
  const [q, setQ] = useState("");
  const go = useNavigate();
  const inDocs = useLocation().pathname.startsWith("/lesson/");
  // The app header is always visible; the Fumadocs sidebar/TOC are offset by its measured height.
  const headerRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = headerRef.current;
    if (!el) return;
    const apply = () => document.documentElement.style.setProperty("--fd-banner-height", `${el.offsetHeight}px`);
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => { ro.disconnect(); document.documentElement.style.removeProperty("--fd-banner-height"); };
  }, []);
  return (
    <FumadocsProvider
      search={{ options: { api: "/api/docs-search" } }}
      i18n={{ locale: lang, locales: [{ name: "English", locale: "en" }, { name: "Français", locale: "fr" }], onLocaleChange: (l) => setLang(l === "fr" ? "fr" : "en"), translations: lang === "fr" ? FUMADOCS_FR : undefined }}
    >
    <div className="min-h-screen">
      <header ref={headerRef} className="sticky top-0 z-40 border-b border-line bg-surface/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-5 gap-y-2 px-4 py-2.5">
          <NavLink to="/" className="text-lg font-bold tracking-tight">ML Academy</NavLink>
          <nav className="flex flex-wrap gap-1 text-sm">
            {nav.map(([to, label]) => (
              <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `rounded-md px-2.5 py-1 ${isActive ? "bg-accentsoft font-semibold text-accent" : "text-muted hover:text-ink"}`}>
                {t(label)}
              </NavLink>
            ))}
          </nav>
          <form className="ml-auto" onSubmit={(e) => { e.preventDefault(); if (q.trim()) go(`/search?q=${encodeURIComponent(q.trim())}`); }}>
            <input className="input w-52 text-sm" placeholder={t("nav.search")} aria-label={t("nav.search")} value={q} onChange={(e) => setQ(e.target.value)} />
          </form>
          <select className="input text-sm" aria-label={t("nav.language")} value={lang} onChange={(e) => setLang(e.target.value === "fr" ? "fr" : "en")}><option value="en">English</option><option value="fr">Français</option></select>
        </div>
      </header>
      <main className={inDocs ? "" : "mx-auto max-w-6xl px-4 py-6"}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/courses" element={<Catalog />} />
          <Route path="/courses/:id" element={<CoursePage />} />
          <Route path="/courses/:id/assessment" element={<AssessmentPage />} />
          <Route element={<DocsShell />}>
            <Route path="/lesson/:id" element={<LessonPage />} />
          </Route>
          <Route path="/concepts" element={<ConceptsPage />} />
          <Route path="/review" element={<ReviewPage />} />
          <Route path="/lab" element={<LabPage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/bookmarks" element={<BookmarksPage />} />
          <Route path="/capstone" element={<CapstonePage />} />
          <Route path="/roadmap" element={<RoadmapPage />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/projects/:id" element={<ProjectPage />} />
          <Route path="/sources" element={<SourcesPage />} />
          <Route path="*" element={<p className="text-muted">{t("common.pageNotFound")}</p>} />
        </Routes>
      </main>
    </div>
    </FumadocsProvider>
  );
}
