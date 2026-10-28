import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import type { AnswerResult, QuestionPublic } from "@shared/types";
import { useApi } from "../api";
import { Question } from "../components/Quiz";
import { useT } from "../i18n";

export function AssessmentPage() {
  const { t } = useT();
  const { id = "" } = useParams();
  const [round, setRound] = useState(0);
  const { data } = useApi<QuestionPublic[]>(`/api/courses/${id}/assessment?n=10&r=${round}`);
  const [score, setScore] = useState({ right: 0, total: 0 });
  if (!data) return <p className="text-muted">{t("common.loading")}</p>;
  return (
    <div className="max-w-3xl space-y-4">
      <Link to={`/courses/${id}`} className="text-sm text-muted hover:underline">{t("assess.back")}</Link>
      <h1 className="text-3xl font-bold tracking-tight">{t("course.assessment")}</h1>
      <p className="text-muted">{t("assess.intro", { n: data.length })}</p>
      <div className="space-y-4" key={round}>{data.map((q, i) => <Question key={q.id} q={q} index={i} onAnswered={(r: AnswerResult) => setScore((s) => ({ right: s.right + (r.correct ? 1 : 0), total: s.total + 1 }))} />)}</div>
      <div className="card flex items-center justify-between p-4">
        <p>{t("assess.score")} <b>{score.right}/{score.total}</b> {t("assess.of", { n: data.length })}</p>
        <button className="btn" onClick={() => { setScore({ right: 0, total: 0 }); setRound((r) => r + 1); }}>{t("assess.new")}</button>
      </div>
    </div>
  );
}
