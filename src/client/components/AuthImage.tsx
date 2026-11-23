import * as React from "react";
import { getApiToken } from "@/lib/api";

export function useAuthImageUrl(path: string | null | undefined): string | null {
  const [objectUrl, setObjectUrl] = React.useState<string | null>(null);
  const token = getApiToken();

  React.useEffect(() => {
    if (!path) {
      setObjectUrl(null);
      return;
    }
    if (!token) {
      setObjectUrl(path);
      return;
    }

    let cancelled = false;
    let createdUrl: string | null = null;
    const controller = new AbortController();

    fetch(path, {
      headers: { Authorization: `Bearer ${token}` },
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.blob();
      })
      .then((blob) => {
        if (cancelled) return;
        createdUrl = URL.createObjectURL(blob);
        setObjectUrl(createdUrl);
      })
      .catch(() => {
        if (!cancelled) setObjectUrl(null);
      });

    return () => {
      cancelled = true;
      controller.abort();
      if (createdUrl) URL.revokeObjectURL(createdUrl);
    };
  }, [path, token]);

  return objectUrl;
}

interface AuthImageProps extends Omit<React.ImgHTMLAttributes<HTMLImageElement>, "src"> {
  src: string;
}

export function AuthImage({ src, alt, ...rest }: AuthImageProps) {
  const resolved = useAuthImageUrl(src);
  if (!resolved) {
    return <div className={rest.className} aria-label={alt} />;
  }
  return <img src={resolved} alt={alt} {...rest} />;
}
