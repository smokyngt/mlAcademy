/** Current UI language, readable outside React (the fetch helper needs it). Updated by LangProvider. */
let current: "en" | "fr" = "en";
export const getLang = () => current;
export const setCurrentLang = (l: "en" | "fr") => { current = l; };
