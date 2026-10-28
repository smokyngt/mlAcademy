import { useMemo, type ComponentProps, type ReactNode } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { FrameworkProvider } from "fumadocs-core/framework";
import { RootProvider } from "fumadocs-ui/provider/base";

/**
 * Fumadocs' ready-made React Router provider requires a *data router* (createBrowserRouter).
 * This app uses the declarative <BrowserRouter>, so we hand Fumadocs the router primitives directly.
 */
function RouterLink({ href = "", prefetch: _prefetch, children, ...rest }: ComponentProps<"a"> & { prefetch?: boolean }) {
  const external = /^(https?:)?\/\//.test(href) || href.startsWith("mailto:");
  if (external || href.startsWith("#") || !href) return <a href={href} {...rest}>{children}</a>;
  return <Link to={href} {...rest}>{children}</Link>;
}

const framework = {
  usePathname: () => useLocation().pathname,
  useParams: () => useParams() as Record<string, string | string[]>,
  useRouter: () => {
    const navigate = useNavigate();
    return useMemo(() => ({ push: (url: string) => navigate(url), refresh: () => {} }), [navigate]);
  },
  Link: RouterLink,
};

export function FumadocsProvider({ children, ...props }: { children: ReactNode } & Omit<ComponentProps<typeof RootProvider>, "children">) {
  return (
    <FrameworkProvider {...framework}>
      <RootProvider {...props}>{children}</RootProvider>
    </FrameworkProvider>
  );
}
