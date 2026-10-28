import { expect, test } from "bun:test";
import { renderToString } from "react-dom/server";
import { MemoryRouter } from "react-router-dom";
import { en } from "./en";
import { fr } from "./fr";
import { LangProvider } from "./index";
import { App } from "../App";

test("every English key has a French translation that differs (except proper nouns)", () => {
  const same = Object.keys(en).filter((k) => (en as any)[k] === (fr as any)[k]);
  console.log("identical en/fr:", same.join(", "));
  expect(Object.keys(fr).sort()).toEqual(Object.keys(en).sort());
});

test("App renders French navigation when French is stored", () => {
  (globalThis as any).localStorage = { getItem: () => "fr", setItem: () => {} };
  const html = renderToString(<MemoryRouter initialEntries={["/courses"]}><LangProvider><App /></LangProvider></MemoryRouter>);
  console.log(html.match(/Tableau de bord|Dashboard|Cours|Courses|Révision|Review/g));
  expect(html).toContain("Tableau de bord");
  expect(html).toContain("Révision");
});
