import { useMemo } from "react";
import { Outlet } from "react-router-dom";
import { DocsLayout } from "fumadocs-ui/layouts/docs";
import type { PathPart } from "@shared/types";
import { useApi } from "../api";
import { buildPathSource } from "./source";
import { useT } from "../i18n";

/** Fumadocs docs layout (sidebar page tree, search, theme switch) around every lesson page. */
export function DocsShell() {
  const { t } = useT();
  const { data } = useApi<PathPart[]>("/api/path");
  const tree = useMemo(() => (data ? buildPathSource(data, { math: t("cat.part.math"), ml: t("cat.part.ml") }, { beginner: t("level.beginner"), intermediate: t("level.intermediate"), advanced: t("level.advanced"), expert: t("level.expert") }).pageTree : null), [data, t]);
  if (!tree) return <p className="p-6 text-muted">{t("common.loading")}</p>;
  return (
    <DocsLayout
      tree={tree}
      sidebar={{ collapsible: false }}
      nav={{ enabled: false }}
    >
      <Outlet />
    </DocsLayout>
  );
}
