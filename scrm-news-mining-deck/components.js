/* ============================================================
   components.js — "공급망 시그널" Deck Design System
   Team G · 지식추론과 텍스트마이닝 Term Project
   Tone: intelligence dossier × editorial science journal
   Fonts: Hahmlet (display) · IBM Plex Sans KR (body) · IBM Plex Mono (data)
   Signature: STRIKE-AND-VERIFY (fake 취소선 → verified teal)
   Export-safe: NO text-shadow / NO small box-shadow blur / NO backdrop-filter
   ============================================================ */

// ───────────────────────────────── Tokens
const TYPE_SCALE = {
  display: 108,  // hero numerals (metric/strike)
  hero: 80,      // secondary big number
  title: 40,     // action title (T1) — identical on every slide
  subtitle: 27,
  body: 24,      // default readable body
  small: 22,     // card descriptions (absolute readable floor)
  label: 15,     // mono eyebrow / meta / page
  source: 13,    // citations only (bottom)
};

const SPACING = {
  paddingTop: 78,
  paddingBottom: 64,
  paddingX: 100,
  railX: 52,
  titleGap: 26,
  itemGap: 18,
};

const C = {
  bg: "#F4F0E7",       // warm ivory canvas (LIGHT theme)
  bgAlt: "#ECE6D8",
  panel: "rgba(31,39,48,0.045)",    // faint dark-tint card on light
  panelAlt: "rgba(31,39,48,0.075)",
  paper: "#FBF8F1",    // brighter paper-variant
  inkOnPaper: "#1F2730",
  paperPanel: "rgba(31,39,48,0.04)",
  ink: "#1F2730",      // dark warm-navy text on light
  inkSoft: "rgba(31,39,48,0.72)",
  inkMuted: "rgba(31,39,48,0.50)",
  rule: "rgba(31,39,48,0.16)",
  ruleSoft: "rgba(31,39,48,0.08)",
  accent: "#BE7A14",   // SIGNAL — deep amber, reads on light
  accentSoft: "rgba(190,122,20,0.13)",
  real: "#2C8A73",     // VERIFIED — deep teal
  realSoft: "rgba(44,138,115,0.13)",
  fake: "#BE3A2B",     // ILLUSION — deep red (struck through)
  fakeSoft: "rgba(190,58,43,0.12)",
};

const FONT_DISPLAY = "'Hahmlet', Georgia, 'Times New Roman', serif";
const FONT_SANS = "'IBM Plex Sans KR', system-ui, -apple-system, sans-serif";
const FONT_MONO = "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace";

const toneColor = (tone) => tone === "real" ? C.real : tone === "fake" ? C.fake
  : tone === "signal" ? C.accent : C.rule;

// ───────────────────────────────── Frame
function SlideFrame({ children, variant = "ink", noChrome = false, glow = true, style = {} }) {
  const paper = variant === "paper";
  const bg = paper ? C.paper : C.bg;
  const ink = paper ? C.inkOnPaper : C.ink;
  const gridLine = "rgba(31,39,48,0.035)";
  const layers = [];
  if (glow) layers.push("radial-gradient(1100px 700px at 86% -8%, rgba(190,122,20,0.06), transparent 62%)");
  layers.push(`repeating-linear-gradient(0deg, transparent 0 79px, ${gridLine} 79px 80px)`);
  layers.push(`repeating-linear-gradient(90deg, transparent 0 79px, ${gridLine} 79px 80px)`);
  return (
    <div style={{
      width: "100%", height: "100%", background: bg, color: ink,
      backgroundImage: layers.join(","),
      fontFamily: FONT_SANS, position: "relative", overflow: "hidden",
      padding: `${SPACING.paddingTop}px ${SPACING.paddingX}px ${SPACING.paddingBottom}px`,
      ...style,
    }}>
      {/* signal-rail: vertical hairline + amber tick in the left gutter */}
      <div style={{ position: "absolute", top: 70, bottom: 56, left: SPACING.railX, width: 1, background: paper ? "rgba(26,34,48,0.16)" : C.rule }} />
      <div style={{ position: "absolute", top: 70, left: SPACING.railX - 3, width: 7, height: 7, background: C.accent }} />
      {!noChrome && <SlideChrome ink={ink} paper={paper} />}
      {children}
    </div>
  );
}

function SlideChrome({ ink, paper }) {
  return (
    <>
      <div style={{ position: "absolute", top: 44, left: SPACING.paddingX, display: "flex", alignItems: "center", gap: 12 }}>
        <BrandMark ink={ink} />
      </div>
      <div style={{
        position: "absolute", top: 48, right: SPACING.paddingX, fontFamily: FONT_MONO,
        fontSize: TYPE_SCALE.label, color: ink, opacity: 0.42, letterSpacing: "0.14em", textTransform: "uppercase",
      }}>지식추론 · 텍스트마이닝 · 2026</div>
    </>
  );
}

function BrandMark({ ink = C.ink, size = 21 }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
      <span style={{ fontFamily: FONT_DISPLAY, fontSize: size, fontWeight: 800, letterSpacing: "0.02em", color: ink }}>TEAM&nbsp;G</span>
      <span style={{ width: 6, height: 6, background: C.accent, display: "inline-block" }} />
      <span style={{ fontFamily: FONT_MONO, fontSize: 13, letterSpacing: "0.18em", color: ink, opacity: 0.55, textTransform: "uppercase" }}>SCRM·SIGNAL</span>
    </div>
  );
}

// ───────────────────────────────── Building blocks
function Eyebrow({ children, color = C.accent, style = {} }) {
  return (
    <div style={{
      fontFamily: FONT_MONO, fontSize: TYPE_SCALE.label, color, letterSpacing: "0.2em",
      textTransform: "uppercase", fontWeight: 500, ...style,
    }}>{children}</div>
  );
}

function SlideTitle({ children, accent, size = TYPE_SCALE.title, style = {} }) {
  return (
    <h1 style={{
      fontFamily: FONT_DISPLAY, fontSize: size, fontWeight: 700, letterSpacing: "-0.015em",
      lineHeight: 1.18, color: "inherit", margin: 0, maxWidth: 1480,
      textWrap: "balance", wordBreak: "keep-all", overflowWrap: "break-word", ...style,
    }}>{children}</h1>
  );
}

function BodyText({ children, size = TYPE_SCALE.body, muted = false, style = {} }) {
  return (
    <p style={{
      fontFamily: FONT_SANS, fontSize: size, fontWeight: 400, lineHeight: 1.5,
      color: muted ? C.inkMuted : C.inkSoft, margin: 0, maxWidth: 1280,
      textWrap: "pretty", wordBreak: "keep-all", overflowWrap: "break-word", ...style,
    }}>{children}</p>
  );
}

function PageNum({ n, total = 14, ink = C.ink }) {
  return (
    <div style={{
      position: "absolute", bottom: 46, right: SPACING.paddingX, fontFamily: FONT_MONO,
      fontSize: TYPE_SCALE.label, color: ink, opacity: 0.42, letterSpacing: "0.1em",
    }}>{String(n).padStart(2, "0")} / {String(total).padStart(2, "0")}</div>
  );
}

function SectionTag({ children, n, ink = C.ink }) {
  return (
    <div style={{
      position: "absolute", bottom: 46, left: SPACING.paddingX, fontFamily: FONT_MONO,
      fontSize: TYPE_SCALE.label, color: ink, opacity: 0.42, letterSpacing: "0.14em", textTransform: "uppercase",
    }}>{children}</div>
  );
}

// ───────────────────────────────── Cards & content
function Card({ children, tone = "default", variant = "ink", style = {} }) {
  const paper = variant === "paper";
  const bar = toneColor(tone);
  const showBar = tone !== "default";
  return (
    <div style={{
      background: paper ? C.paperPanel : C.panel,
      borderLeft: showBar ? `3px solid ${bar}` : `1px solid ${paper ? "rgba(26,34,48,0.12)" : C.rule}`,
      border: showBar ? undefined : (paper ? "1px solid rgba(26,34,48,0.12)" : `1px solid ${C.rule}`),
      borderRadius: 10, padding: "24px 26px", color: "inherit", ...style,
    }}>{children}</div>
  );
}

function CardTitle({ children, tone, style = {} }) {
  return <div style={{ fontFamily: FONT_SANS, fontSize: TYPE_SCALE.subtitle, fontWeight: 600, color: tone ? toneColor(tone) : "inherit", letterSpacing: "-0.01em", lineHeight: 1.2, marginBottom: 10, ...style }}>{children}</div>;
}

function CardBody({ children, style = {} }) {
  return <div style={{ fontFamily: FONT_SANS, fontSize: TYPE_SCALE.small, fontWeight: 400, lineHeight: 1.46, color: C.inkSoft, ...style }}>{children}</div>;
}

function Callout({ children, tone = "signal", variant = "ink", style = {} }) {
  const col = toneColor(tone);
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 16, padding: "18px 24px",
      background: tone === "real" ? C.realSoft : tone === "fake" ? C.fakeSoft : C.accentSoft,
      borderLeft: `3px solid ${col}`, borderRadius: "0 10px 10px 0", ...style,
    }}>
      <span style={{ fontFamily: FONT_MONO, fontSize: TYPE_SCALE.label, color: col, letterSpacing: "0.14em", whiteSpace: "nowrap" }}>→ 판정</span>
      <span style={{ fontFamily: FONT_SANS, fontSize: TYPE_SCALE.body, fontWeight: 500, color: variant === "paper" ? C.inkOnPaper : C.ink, lineHeight: 1.4 }}>{children}</span>
    </div>
  );
}

// ───────────────────────────────── Metric / Strike (signature)
function MetricTile({ value, unit, label, sub, tone = "signal", style = {} }) {
  const col = toneColor(tone);
  return (
    <div style={{ position: "relative", ...style }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <span style={{ fontFamily: FONT_DISPLAY, fontSize: TYPE_SCALE.display, fontWeight: 800, color: col, lineHeight: 0.92, letterSpacing: "-0.02em" }}>{value}</span>
        {unit && <span style={{ fontFamily: FONT_MONO, fontSize: 26, color: col, opacity: 0.85 }}>{unit}</span>}
      </div>
      {label && <div style={{ fontFamily: FONT_SANS, fontSize: TYPE_SCALE.body, fontWeight: 600, marginTop: 12, lineHeight: 1.3, color: C.ink }}>{label}</div>}
      {sub && <div style={{ fontFamily: FONT_MONO, fontSize: TYPE_SCALE.label, color: C.inkMuted, marginTop: 6, letterSpacing: "0.04em" }}>{sub}</div>}
    </div>
  );
}

// STRIKE-AND-VERIFY — the deck's signature: fake (struck) → verified
function StrikeStat({ fake, fakeLabel, real, realLabel, note, style = {} }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 40, flexWrap: "wrap", ...style }}>
      <div>
        <div style={{ fontFamily: FONT_DISPLAY, fontSize: TYPE_SCALE.display, fontWeight: 800, color: C.fake, lineHeight: 0.92, textDecoration: "line-through", textDecorationThickness: 4 }}>{fake}</div>
        <div style={{ fontFamily: FONT_MONO, fontSize: TYPE_SCALE.label, color: C.fake, marginTop: 10, letterSpacing: "0.1em", maxWidth: 240, lineHeight: 1.4 }}>{fakeLabel}</div>
      </div>
      <div style={{ fontFamily: FONT_MONO, fontSize: 40, color: C.inkMuted }}>→</div>
      <div>
        <div style={{ fontFamily: FONT_DISPLAY, fontSize: TYPE_SCALE.display, fontWeight: 800, color: C.real, lineHeight: 0.92 }}>{real}</div>
        <div style={{ fontFamily: FONT_MONO, fontSize: TYPE_SCALE.label, color: C.real, marginTop: 10, letterSpacing: "0.1em", maxWidth: 240, lineHeight: 1.4 }}>✓ {realLabel}</div>
      </div>
      {note && <div style={{ fontFamily: FONT_SANS, fontSize: TYPE_SCALE.small, color: C.inkSoft, maxWidth: 360, lineHeight: 1.45, borderLeft: `1px solid ${C.rule}`, paddingLeft: 20 }}>{note}</div>}
    </div>
  );
}

// ───────────────────────────────── Comparison column
function ComparisonColumn({ tag, title, tone = "default", items = [], style = {} }) {
  const col = toneColor(tone);
  return (
    <div style={{ flex: 1, background: C.panel, borderTop: `3px solid ${tone === "default" ? C.rule : col}`, borderRadius: "0 0 10px 10px", padding: "22px 26px 24px", ...style }}>
      {tag && <div style={{ fontFamily: FONT_MONO, fontSize: TYPE_SCALE.label, color: col, letterSpacing: "0.16em", textTransform: "uppercase", marginBottom: 6 }}>{tag}</div>}
      <div style={{ fontFamily: FONT_DISPLAY, fontSize: TYPE_SCALE.subtitle, fontWeight: 700, color: C.ink, marginBottom: 16, lineHeight: 1.2 }}>{title}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {items.map((it, i) => (
          <div key={i} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
            <span style={{ width: 6, height: 6, background: col, marginTop: 11, flex: "0 0 auto" }} />
            <span style={{ fontFamily: FONT_SANS, fontSize: TYPE_SCALE.small, color: C.inkSoft, lineHeight: 1.42 }}>{it}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ───────────────────────────────── Process row (steps)
function ProcessRow({ steps = [], style = {} }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${steps.length}, 1fr)`, gap: 12, alignItems: "stretch", ...style }}>
      {steps.map((s, i) => (
        <div key={i} style={{ background: C.panel, border: `1px solid ${C.rule}`, borderRadius: 10, padding: "18px 16px", position: "relative" }}>
          <div style={{ fontFamily: FONT_MONO, fontSize: TYPE_SCALE.label, color: C.accent, letterSpacing: "0.1em" }}>{String(i + 1).padStart(2, "0")}</div>
          <div style={{ fontFamily: FONT_SANS, fontSize: 21, fontWeight: 600, color: C.ink, marginTop: 8, lineHeight: 1.2 }}>{s.t}</div>
          <div style={{ fontFamily: FONT_SANS, fontSize: 17, color: C.inkMuted, marginTop: 8, lineHeight: 1.4 }}>{s.d}</div>
        </div>
      ))}
    </div>
  );
}

// ───────────────────────────────── Quote
function Quote({ children, attr, variant = "ink", style = {} }) {
  return (
    <div style={{ ...style }}>
      <div style={{ fontFamily: FONT_DISPLAY, fontSize: TYPE_SCALE.subtitle, fontWeight: 500, fontStyle: "italic", lineHeight: 1.4, color: variant === "paper" ? C.inkOnPaper : C.ink, borderLeft: `3px solid ${C.accent}`, paddingLeft: 24 }}>
        “{children}”
      </div>
      {attr && <div style={{ fontFamily: FONT_MONO, fontSize: TYPE_SCALE.label, color: C.inkMuted, marginTop: 14, paddingLeft: 27, letterSpacing: "0.06em" }}>— {attr}</div>}
    </div>
  );
}

// ───────────────────────────────── Bar (lightweight, data-ink max)
function BarRow({ label, value, max = 1, color = C.accent, display, baseline = false, style = {} }) {
  const pct = Math.max(2, Math.min(100, (value / max) * 100));
  return (
    <div style={{ display: "grid", gridTemplateColumns: "240px 1fr 110px", alignItems: "center", gap: 18, ...style }}>
      <div style={{ fontFamily: FONT_SANS, fontSize: TYPE_SCALE.small, color: C.inkSoft, textAlign: "right" }}>{label}</div>
      <div style={{ position: "relative", height: 30, background: C.ruleSoft, borderRadius: 4 }}>
        <div style={{ position: "absolute", top: 0, left: 0, height: "100%", width: `${pct}%`, background: color, borderRadius: 4, opacity: baseline ? 0.42 : 1 }} />
      </div>
      <div style={{ fontFamily: FONT_MONO, fontSize: 22, fontWeight: 600, color, textAlign: "right" }}>{display}</div>
    </div>
  );
}

function SourceLine({ children, ink = C.inkMuted, style = {} }) {
  return (
    <div style={{
      position: "absolute", bottom: 46, left: SPACING.paddingX, maxWidth: 1100,
      fontFamily: FONT_MONO, fontSize: TYPE_SCALE.source, color: ink, opacity: 0.7, letterSpacing: "0.02em", ...style,
    }}>출처 · {children}</div>
  );
}

function Placeholder({ label, style = {} }) {
  return <div style={{ border: `1px dashed ${C.rule}`, borderRadius: 8, padding: 40, fontFamily: FONT_MONO, color: C.inkMuted, textAlign: "center", ...style }}>{label}</div>;
}

// ───────────────────────────────── Global exposure
Object.assign(window, {
  TYPE_SCALE, SPACING, C, FONT_DISPLAY, FONT_SANS, FONT_MONO, toneColor,
  SlideFrame, SlideChrome, BrandMark,
  Eyebrow, SlideTitle, BodyText, PageNum, SectionTag,
  Card, CardTitle, CardBody, Callout,
  MetricTile, StrikeStat, ComparisonColumn, ProcessRow, Quote, BarRow, SourceLine, Placeholder,
});
