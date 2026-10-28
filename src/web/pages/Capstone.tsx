import { useState } from "react";
import { api, useApi } from "../api";
import { useT } from "../i18n";

interface Criterion { id: string; text: string; evidence: string; latest: { text: string; selfRating: number; at: number } | null }
interface Cap { id: string; title: string; brief: string; criteria: Criterion[] }
const RATINGS = ["capstone.r0", "capstone.r1", "capstone.r2", "capstone.r3"] as const;

function CriterionForm({ capId, c, onSaved }: { capId: string; c: Criterion; onSaved: () => void }) {
  const { t, lang } = useT();
  const [text, setText] = useState(c.latest?.text ?? "");
  const [rating, setRating] = useState(c.latest?.selfRating ?? 0);
  const [err, setErr] = useState("");
  async function save() {
    setErr("");
    try { await api(`/api/capstones/${capId}/evidence`, { body: { criterion: c.id, text, selfRating: rating } }); onSaved(); }
    catch (e) { setErr(String((e as Error).message)); }
  }
  return (
    <div className="card p-4">
      <p className="font-semibold">{c.text}</p>
      <p className="mb-2 text-sm text-muted">{t("capstone.evidence", { e: c.evidence })}</p>
      <textarea className="input w-full" rows={6} value={text} onChange={(e) => setText(e.target.value)} aria-label={t("capstone.evidenceFor", { c: c.text })} />
      <div className="mt-2 flex flex-wrap items-center gap-3 text-sm">
        <span className="text-muted">{t("capstone.self")}</span>
        {RATINGS.map((r, i) => <label key={r} className="flex items-center gap-1"><input type="radio" name={c.id} checked={rating === i} onChange={() => setRating(i)} />{t(r)}</label>)}
        <button className="btn btn-primary ml-auto" onClick={save}>{t("capstone.save")}</button>
      </div>
      {err && <p className="mt-1 text-sm text-bad">{err}</p>}
      {c.latest && <p className="mt-1 text-xs text-muted">{t("capstone.lastSaved", { when: new Date(c.latest.at).toLocaleString(lang), rating: t(RATINGS[c.latest.selfRating]) })}</p>}
    </div>
  );
}

export function CapstonePage() {
  const { t } = useT();
  const { data, reload } = useApi<Cap[]>("/api/capstones");
  if (!data) return <p className="text-muted">{t("common.loading")}</p>;
  return (
    <div className="max-w-3xl space-y-8">
      <div><h1 className="text-3xl font-bold tracking-tight">{t("capstone.title")}</h1><p className="text-muted">{t("capstone.intro")}</p></div>
      {data.map((cap) => {
        const done = cap.criteria.filter((c) => c.latest && c.latest.selfRating >= 2).length;
        return (
          <section key={cap.id} className="space-y-4">
            <h2 className="text-2xl font-bold">{cap.title}</h2>
            <p>{cap.brief}</p>
            <p className="text-sm text-muted">{t("capstone.criteria", { done, n: cap.criteria.length })}</p>
            {cap.criteria.map((c) => <CriterionForm key={c.id + (c.latest?.at ?? 0)} capId={cap.id} c={c} onSaved={reload} />)}
          </section>
        );
      })}
    </div>
  );
}
