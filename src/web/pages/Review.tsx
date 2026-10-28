import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import type { Card } from "@shared/types";
import { api } from "../api";
import { Inline } from "../components/Markdown";
import { useT, type Key } from "../i18n";

const GRADES: { g: number; label: Key; cls: string }[] = [
  { g: 0, label: "rev.again", cls: "!border-bad text-bad" },
  { g: 1, label: "rev.hard", cls: "!border-warn text-warn" },
  { g: 2, label: "rev.good", cls: "!border-accent text-accent" },
  { g: 3, label: "rev.easy", cls: "!border-good text-good" },
];

const fmtDays = (t: ReturnType<typeof useT>["t"], d: number) => (d < 0.01 ? t("rev.in.min") : d < 1 ? t("rev.in.h", { n: Math.round(d * 24) }) : t("rev.in.d", { n: Math.round(d) }));

export function ReviewPage() {
  const { t } = useT();
  const [queue, setQueue] = useState<Card[] | null>(null);
  const [shown, setShown] = useState(false);
  const [done, setDone] = useState(0);
  const [last, setLast] = useState("");

  const load = () => api<{ cards: Card[] }>("/api/review/session?new=10").then((r) => { setQueue(r.cards); setDone(0); });
  useEffect(() => { load(); }, []);

  if (!queue) return <p className="text-muted">{t("common.loading")}</p>;
  const card = queue[0];

  async function grade(g: number) {
    const r = await api<{ intervalDays: number }>(`/api/review/${card.id}`, { body: { grade: g } });
    setLast(t("rev.next", { when: fmtDays(t, r.intervalDays) }));
    setShown(false);
    setDone((n) => n + 1);
    // "Again" cards return to the back of this session; everything else leaves it.
    setQueue((q) => (g === 0 ? [...q!.slice(1), q![0]] : q!.slice(1)));
  }

  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <div><h1 className="text-3xl font-bold tracking-tight">{t("rev.title")}</h1><p className="text-muted">{t("rev.subtitle")}</p></div>
      {!card ? (
        <div className="card p-8 text-center">
          <p className="text-xl font-semibold">{done > 0 ? t("rev.complete", { n: done }) : t("rev.nothing")}</p>
          <p className="mt-1 text-muted">{t("rev.nothingHelp")}</p>
          <div className="mt-4 flex justify-center gap-2"><button className="btn" onClick={load}>{t("rev.check")}</button><Link className="btn btn-primary" to="/courses">{t("rev.back")}</Link></div>
        </div>
      ) : (
        <div className="card p-6">
          <div className="mb-4 flex items-center gap-2 text-xs text-muted"><span className="chip">{card.conceptTitle}</span>{card.isNew && <span className="chip !border-accent !text-accent">{t("rev.new")}</span>}<span className="ml-auto">{t("rev.left", { n: queue.length })}</span></div>
          <p className="min-h-20 text-xl"><Inline source={card.front} /></p>
          {shown ? (
            <>
              <hr className="my-4 border-line" />
              <p className="text-lg"><Inline source={card.back} /></p>
              <div className="mt-6 grid grid-cols-4 gap-2">{GRADES.map((x) => <button key={x.g} className={`btn justify-center ${x.cls}`} onClick={() => grade(x.g)}>{t(x.label)}</button>)}</div>
            </>
          ) : (
            <button className="btn btn-primary mt-6" onClick={() => setShown(true)}>{t("rev.show")}</button>
          )}
          {card.lessonId && <p className="mt-4 text-xs text-muted">{t("rev.source")} <Link className="text-accent underline" to={`/lesson/${card.lessonId}`}>{card.lessonId}</Link></p>}
        </div>
      )}
      {last && <p className="text-center text-sm text-muted" role="status">{last}</p>}
    </div>
  );
}
