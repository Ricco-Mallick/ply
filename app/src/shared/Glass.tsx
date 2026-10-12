import { useEffect, useRef, type ReactNode } from 'react';
import { attachGlass, type GlassOptions } from './glass';

/** A liquid-glass panel (see glass.ts). `as` picks the element. */
export function Glass({ as: Tag = 'div', opts, className = '', children, ...rest }: { as?: any; opts?: GlassOptions; className?: string; children?: ReactNode; [k: string]: any }) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => (ref.current ? attachGlass(ref.current, opts) : undefined), []);
  return <Tag ref={ref} className={`lg ${className}`} {...rest}>{children}</Tag>;
}
