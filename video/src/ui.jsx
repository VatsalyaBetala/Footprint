// Shared pieces, styled after frontend/index.html (same colour tokens and fonts).
import { AbsoluteFill, Img, interpolate, spring, staticFile } from "remotion";
import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadMono } from "@remotion/google-fonts/JetBrainsMono";

const { fontFamily: SANS } = loadInter("normal", { weights: ["400", "500", "600", "700"], subsets: ["latin"] });
const { fontFamily: MONO } = loadMono("normal", { weights: ["500", "700"], subsets: ["latin"] });
export { SANS, MONO };

export const C = {
  bg: "#f5f6f8", panel: "#ffffff", ink: "#0f172a", muted: "#5b6577", line: "#e3e7ee", lineStrong: "#cbd3df",
  accent: "#2563eb", accentTint: "#eaf0ff", accentInk: "#1d4ed8",
  ok: "#10b981", okInk: "#047857", okTint: "#e7f8f1", warn: "#f59e0b", warnInk: "#b45309", warnTint: "#fff5e1",
  bad: "#ef4444", badInk: "#b91c1c", badTint: "#fdecec",
};
export const CLAMP = { extrapolateLeft: "clamp", extrapolateRight: "clamp" };

/** 0..1 ease-in over [start, start+len] */
export const ramp = (f, start, len = 15) => interpolate(f, [start, start + len], [0, 1], CLAMP);
/** spring 0..1 that starts at `start` */
export const pop = (f, fps, start = 0, damping = 200) => spring({ frame: f - start, fps, config: { damping } });
/** scene-level fade in + out */
export const sceneFade = (f, dur, inLen = 10, outLen = 10) =>
  Math.min(interpolate(f, [0, inLen], [0, 1], CLAMP), interpolate(f, [dur - outLen, dur], [1, 0], CLAMP));

export const Page = ({ children, style }) => (
  <AbsoluteFill style={{ background: C.bg, fontFamily: SANS, color: C.ink, ...style }}>{children}</AbsoluteFill>
);

/** faint blueprint grid behind type-only scenes */
export const Grid = ({ opacity = 1 }) => (
  <AbsoluteFill style={{
    opacity,
    backgroundImage: `linear-gradient(${C.line} 1px, transparent 1px), linear-gradient(90deg, ${C.line} 1px, transparent 1px)`,
    backgroundSize: "48px 48px",
  }} />
);

/** app window: header with logo + live pill, then content */
export const AppFrame = ({ children, title = "Footprint", right = "backend live", style }) => (
  <div style={{
    position: "absolute", left: 80, top: 70, width: 1760, height: 940, background: C.panel, borderRadius: 18,
    boxShadow: "0 18px 60px rgba(15,23,42,.12)", border: `1px solid ${C.line}`, overflow: "hidden", ...style,
  }}>
    <div style={{ height: 64, display: "flex", alignItems: "center", gap: 14, padding: "0 28px", borderBottom: `1px solid ${C.line}` }}>
      <Logo size={26} />
      <span style={{ fontWeight: 700, fontSize: 24 }}>{title}</span>
      <span style={{ marginLeft: "auto", fontSize: 16, fontWeight: 600, color: C.okInk, background: C.okTint, padding: "6px 14px", borderRadius: 999 }}>
        ● {right}
      </span>
    </div>
    <div style={{ position: "relative", height: 876 }}>{children}</div>
  </div>
);

export const Logo = ({ size = 40 }) => (
  <svg width={size} height={size} viewBox="0 0 40 40">
    <rect x="3" y="3" width="34" height="34" rx="8" fill={C.accent} />
    <path d="M11 29V11h18v8H19v10" fill="none" stroke="#fff" strokeWidth="3.4" strokeLinejoin="round" />
  </svg>
);

export const Chip = ({ kind = "ok", children, style }) => {
  const m = { ok: [C.okTint, C.okInk], warn: [C.warnTint, C.warnInk], bad: [C.badTint, C.badInk], accent: [C.accentTint, C.accentInk], plain: ["#eef1f5", C.muted] }[kind];
  return (
    <span style={{ background: m[0], color: m[1], fontWeight: 700, fontSize: 15, letterSpacing: ".06em", padding: "6px 12px", borderRadius: 8, fontFamily: MONO, ...style }}>
      {children}
    </span>
  );
};

/** answer tile like the app's: question, big answer, detail; slides up with `p` (0..1) */
export const Tile = ({ kind, q, a, sub, p = 1, style }) => {
  const bar = { ok: C.ok, warn: C.warn, bad: C.bad, none: C.lineStrong }[kind];
  return (
    <div style={{
      background: C.panel, border: `1px solid ${C.line}`, borderLeft: `6px solid ${bar}`, borderRadius: 14, padding: "22px 26px",
      boxShadow: "0 4px 18px rgba(15,23,42,.07)", opacity: p, transform: `translateY(${(1 - p) * 30}px)`, ...style,
    }}>
      <div style={{ fontSize: 18, color: C.muted, fontWeight: 600 }}>{q}</div>
      <div style={{ fontSize: 34, fontWeight: 700, marginTop: 6 }}>{a}</div>
      <div style={{ fontSize: 19, color: C.muted, marginTop: 8, lineHeight: 1.35 }}>{sub}</div>
    </div>
  );
};

/** plan image with an SVG overlay in image-pixel coordinates */
export const PlanView = ({ src, w, h, width, children, style }) => (
  <div style={{ position: "absolute", width, height: (width * h) / w, ...style }}>
    <Img src={staticFile(src)} style={{ width: "100%", height: "100%", display: "block" }} />
    <svg viewBox={`0 0 ${w} ${h}`} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible" }}>
      {children}
    </svg>
  </div>
);

export const perim = (p) => p.reduce((L, a, i) => { const b = p[(i + 1) % p.length]; return L + Math.hypot(b[0] - a[0], b[1] - a[1]); }, 0);
export const centroid = (p) => [p.reduce((s, q) => s + q[0], 0) / p.length, p.reduce((s, q) => s + q[1], 0) / p.length];
export const pts = (p) => p.map((q) => q.join(",")).join(" ");

/** outline that draws itself on (draw 0..1), then fills */
export const DrawPoly = ({ poly, color, draw = 1, fill = 0, sw = 2.4, dash }) => {
  const L = perim(poly);
  return (
    <polygon points={pts(poly)} fill={color} fillOpacity={fill} stroke={color} strokeWidth={sw} strokeLinejoin="round"
      strokeDasharray={dash || L} strokeDashoffset={dash ? 0 : L * (1 - draw)} />
  );
};

/** "12.5" ft -> 12'6" */
export const ftin = (ft) => { const i = Math.round(ft * 12); return `${Math.floor(i / 12)}'${i % 12}"`; };
