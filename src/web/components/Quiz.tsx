import { useState } from "react";
import { Link } from "react-router-dom";
import type { AnswerResult, QuestionPublic } from "@shared/types";
import { api } from "../api";
import { Inline } from "./Markdown";
import { useT } from "../i18n";

export function Question({ q, index, onAnswered }: { q: QuestionPublic; index: number; onAnswered?: (r: AnswerResult) => void }) {
  const { t } = useT();
  const [choice, setChoice] = useState<number | null>(null);
  const [num, setNum] = useState("");
  const [result, setResult] = useState<AnswerResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function submit() {
    const response = q.kind === "numeric" ? Number(num) : choice;
    if (response === null || (q.kind === "numeric" && (num.trim() === "" || Number.isNaN(response)))) return;
    setBusy(true); setErr("");
    try {
      const r = await api<AnswerResult>(`/api/questions/${q.id}/answer`, { body: { response } });
      setResult(r);
      onAnswered?.(r);
    } catch (e) { setErr(String((e as Error).message)); }
    setBusy(false);
  }
  const retry = () => { setResult(null); setChoice(null); setNum(""); };

  return (
    <div className="card p-4">
      <p className="mb-3 font-medium"><span className="mr-2 text-muted">{index + 1}.</span><Inline source={q.prompt} /></p>
      {q.kind === "mcq" ? (
        <div className="space-y-2" role="radiogroup" aria-label={t("quiz.questionAria", { n: index + 1 })}>
          {q.options.map((o, i) => {
            const state = result ? (i === result.answer ? "right" : i === choice ? "wrong" : "") : "";
            return (
              <label key={i} className={`flex cursor-pointer items-start gap-2 rounded-lg border px-3 py-2 text-sm ${state === "right" ? "border-good bg-good/10" : state === "wrong" ? "border-bad bg-bad/10" : choice === i ? "border-accent bg-accentsoft" : "border-line hover:bg-surface2"}`}>
                <input type="radio" className="mt-1" name={q.id} checked={choice === i} disabled={!!result} onChange={() => setChoice(i)} />
                <span><Inline source={o} /></span>
              </label>
            );
          })}
        </div>
      ) : (
        <input className="input w-48 font-mono" inputMode="decimal" placeholder={t("quiz.placeholder")} aria-label={t("quiz.numericAria")} value={num} disabled={!!result} onChange={(e) => setNum(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit()} />
      )}
      <div className="mt-3 flex items-center gap-3">
        {!result ? <button className="btn btn-primary" disabled={busy || (q.kind === "mcq" ? choice === null : num.trim() === "")} onClick={submit}>{t("quiz.check")}</button> : <button className="btn" onClick={retry}>{t("quiz.retry")}</button>}
        {err && <span className="text-sm text-bad">{err}</span>}
      </div>
      {result && (
        <div className={`mt-3 rounded-lg border p-3 text-sm ${result.correct ? "border-good/40 bg-good/10" : "border-bad/40 bg-bad/10"}`} role="status">
          <p className="font-semibold">{result.correct ? t("quiz.correct") : q.kind === "numeric" ? t("quiz.wrongNumeric", { a: result.answer }) : t("quiz.wrong")}</p>
          <p className="mt-1"><Inline source={result.explanation} /></p>
          {result.recommendation && (
            <div className="mt-3 rounded-md border border-warn/50 bg-warn/10 p-2">
              <p className="font-semibold text-warn">{t("quiz.reviewTitle")}</p>
              <p>{t("quiz.reviewBody", { reason: result.recommendation.reason, concept: result.recommendation.conceptTitle })}</p>
              <ul className="mt-1 list-disc pl-5">
                {result.recommendation.review.length === 0 && <li>{t("quiz.noWeak")}</li>}
                {result.recommendation.review.map((r) => (
                  <li key={r.conceptId}>{r.lessonId ? <Link className="text-accent underline" to={`/lesson/${r.lessonId}`}>{r.title}</Link> : r.title} <span className="text-muted">({r.score === null ? t("common.notAttempted") : t("quiz.mastery", { pct: Math.round(r.score * 100) })})</span></li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
