import * as React from "react";
import { cn } from "@/lib/utils";

export const Select = React.forwardRef<HTMLSelectElement, React.SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, children, ...props }, ref) => {
    const safeProps =
      "value" in props
        ? { ...props, value: props.value === undefined || props.value === null ? "" : props.value }
        : props;
    return (
      <select
        ref={ref}
        className={cn(
          "flex h-10 w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/60 disabled:opacity-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-100",
          className
        )}
        {...safeProps}
      >
        {children}
      </select>
    );
  }
);
Select.displayName = "Select";
