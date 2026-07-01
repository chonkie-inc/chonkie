import type { ReactNode } from "react";

interface ParamFieldProps {
  path?: string;
  body?: string;
  query?: string;
  type?: string;
  default?: string;
  required?: boolean;
  children?: ReactNode;
}

export function ParamField({
  path,
  body,
  query,
  type,
  default: defaultValue,
  required,
  children,
}: ParamFieldProps) {
  const name = path || body || query;

  return (
    <div className="my-4 rounded-lg border border-fd-border bg-fd-card p-4">
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <code className="text-sm font-semibold text-fd-primary">{name}</code>
        {required && (
          <span className="text-xs font-medium text-red-500 bg-red-500/10 px-1.5 py-0.5 rounded">
            Required
          </span>
        )}
        {type && (
          <span className="text-xs text-fd-muted-foreground bg-fd-muted px-2 py-0.5 rounded">
            {type}
          </span>
        )}
        {defaultValue && (
          <span className="text-xs text-fd-muted-foreground">
            Default: <code className="text-xs">{defaultValue}</code>
          </span>
        )}
      </div>
      {children && (
        <div className="text-sm text-fd-muted-foreground">{children}</div>
      )}
    </div>
  );
}
