import * as React from "react";
import { cn } from "@/lib/utils";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => {
    const safeProps =
      "value" in props
        ? {
            ...props,
            value:
              props.value === undefined ||
              props.value === null ||
              (typeof props.value === "number" && Number.isNaN(props.value))
                ? ""
                : props.value,
          }
        : props;
    return (
      <input
        ref={ref}
        className={cn(
          "flex h-10 w-full rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm text-zinc-900 placeholder:text-zinc-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/60 disabled:opacity-50 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-100 dark:placeholder:text-zinc-500",
          className
        )}
        {...safeProps}
      />
    );
  }
);
Input.displayName = "Input";
