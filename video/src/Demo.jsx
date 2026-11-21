// Footprint demo, ~43 s. Every number on screen comes from a real run exported by scratch/video_data.py
// (plan1 geometry + live verify, solver fits on the synthetic plan, the hotel-room /scan) or from README results.
import { AbsoluteFill, Img, Series, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import data from "./data.json";
import {
  AppFrame, C, CLAMP, Chip, DrawPoly, Grid, Logo, MONO, Page, PlanView, Tile, centroid, ftin, pop, pts, ramp, sceneFade,
} from "./ui";

const D = { hook: 120, plan: 240, answers: 210, fit: 210, scan: 270, proof: 165, end: 90 };
export const TOTAL = Object.values(D).reduce((a, b) => a + b, 0);

const P1 = data.plan1;
const PLAN_W = 1060;
const statusColor = (r) => (r.verified ? C.ok : C.warn);
const Caption = ({ children, style }) => (
  <div style={{ position: "absolute", fontFamily: MONO, fontSize: 15, color: C.muted, ...style }}>{children}</div>
);

// ---------- 1. hook ----------
const Hook = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const a = pop(f, fps, 6), b = pop(f, fps, 28), c = ramp(f, 60, 18);
  const big = { fontSize: 128, fontWeight: 700, letterSpacing: "-0.035em", lineHeight: 1.05 };
  return (
    <Page>
      <Grid opacity={0.55} />
      <AbsoluteFill style={{ opacity: sceneFade(f, D.hook, 1, 12), justifyContent: "center", alignItems: "center" }}>
        <div style={{ ...big, opacity: a, transform: `translateY(${(1 - a) * 50}px)` }}>Is that 2BR</div>
        <div style={{ ...big, color: C.accent, opacity: b, transform: `translateY(${(1 - b) * 50}px)` }}>actually a 2BR?</div>
        <div style={{ marginTop: 46, display: "flex", alignItems: "center", gap: 16, opacity: c, fontSize: 32, color: C.muted }}>
          <Logo size={46} /> <b style={{ color: C.ink }}>Footprint</b> checks an apartment listing against reality.
        </div>
      </AbsoluteFill>
    </Page>
  );
};

// ---------- 2. upload plan: trace + verify ----------
const Step = ({ n, text, sub, start, done, f }) => {
  const p = ramp(f, start, 12), d = f >= done;
  return (
    <div style={{ opacity: p, transform: `translateX(${(1 - p) * 24}px)`, display: "flex", gap: 18, marginBottom: 34 }}>
      <div style={{
        width: 44, height: 44, borderRadius: 22, flex: "none", display: "flex", alignItems: "center", justifyContent: "center",
        background: d ? C.ok : C.accentTint, color: d ? "#fff" : C.accentInk, fontWeight: 700, fontSize: 21,
      }}>{d ? "✓" : n}</div>
      <div>
        <div style={{ fontSize: 27, fontWeight: 600, lineHeight: 1.25 }}>{text}</div>
        {sub && <div style={{ fontSize: 21, color: C.muted, marginTop: 6, lineHeight: 1.3 }}>{sub}</div>}
      </div>
    </div>
  );
};

const RoomLayer = ({ f, fps, traceStart, verifyStart, highlight }) => (
  <>
    {P1.rooms.map((r, i) => {
      const draw = ramp(f, traceStart + i * 14, 22);
      const vt = verifyStart + i * 7;
      const verified = f >= vt;
      const col = verified ? statusColor(r) : C.accent;
      const [cx, cy] = centroid(r.polygon);
      const b = pop(f, fps, vt, 14);
      const hl = highlight === r.id ? 0.5 + 0.5 * Math.sin(f / 5) : 0;
      return (
        <g key={r.id}>
          <DrawPoly poly={r.polygon} color={col} draw={draw} fill={verified ? 0.16 + hl * 0.2 : 0} sw={highlight === r.id ? 4 : 2.6} />
          {verified && (
            <g transform={`translate(${cx},${cy}) scale(${b})`}>
              <circle r="13" fill={col} stroke="#fff" strokeWidth="2.5" />
              <text textAnchor="middle" dy="5" fontSize="15" fontWeight="700" fill="#fff">{r.verified ? "✓" : "~"}</text>
            </g>
          )}
        </g>
      );
    })}
  </>
);

const PlanScene = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const nOk = P1.rooms.filter((r) => r.verified).length;
  const traceStart = 50, verifyStart = 168;
  return (
    <Page style={{ opacity: sceneFade(f, D.plan) }}>
      <AppFrame>
        <PlanView src="plan1.png" w={P1.w} h={P1.h} width={PLAN_W} style={{ left: 36, top: 30, opacity: ramp(f, 0, 15) }}>
          <RoomLayer f={f} fps={fps} traceStart={traceStart} verifyStart={verifyStart} />
        </PlanView>
        <Caption style={{ left: 40, top: 30 + (PLAN_W * P1.h) / P1.w + 8 }}>real listing plan · samples/plan1.png</Caption>
        <div style={{ position: "absolute", left: 1150, top: 60, width: 560 }}>
          <div style={{ fontSize: 18, fontWeight: 700, letterSpacing: ".08em", color: C.muted, marginBottom: 30, opacity: ramp(f, 4) }}>
            CHECKING THE LISTING
          </div>
          <Step f={f} n={1} start={12} done={46} text="Gemini 4 Carbon reads labels, printed sizes and doors" />
          <Step f={f} n={2} start={50} done={traceStart + 6 * 14 + 22} text="OpenCV traces every room's walls"
            sub="Models never draw coordinates. Geometry is computed." />
          <Step f={f} n={3} start={verifyStart - 6} done={verifyStart + 7 * 7 + 8}
            text="Gemini Decisions checks each outline against the image"
            sub={f >= verifyStart + 7 * 7 + 8 ? `${nOk} confirmed · ${P1.rooms.length - nOk} marked estimated, not hidden` : "…"} />
        </div>
      </AppFrame>
    </Page>
  );
};

// ---------- 3. answers ----------
const AnswersScene = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const bed = P1.legal[0];
  const bedRoom = P1.rooms.find((r) => r.id === bed.room);
  const area = P1.area;
  return (
    <Page style={{ opacity: sceneFade(f, D.answers) }}>
      <AppFrame>
        <PlanView src="plan1.png" w={P1.w} h={P1.h} width={PLAN_W} style={{ left: 36, top: 30 }}>
          <RoomLayer f={1000} fps={fps} traceStart={-1000} verifyStart={-1000} highlight={f > 15 ? bedRoom.id : null} />
        </PlanView>
        <div style={{ position: "absolute", left: 1150, top: 46, width: 570, display: "flex", flexDirection: "column", gap: 22 }}>
          <div style={{ opacity: ramp(f, 4), fontSize: 18, fontWeight: 700, letterSpacing: ".08em", color: C.muted }}>THE ANSWERS</div>
          <Tile p={pop(f, fps, 15)} kind="ok" q="Is it really a 1-bedroom?" a="Yes: 1 legal bedroom"
            sub={`${bedRoom.label}: ${Math.round(bed.measurements.printed_sqft)} sq ft, ${bedRoom.printed_dims_ft ? Math.min(...bedRoom.printed_dims_ft) : ""} ft wide. NYC minimum: 80 sq ft and 8 ft.`} />
          <Tile p={pop(f, fps, 60)} kind="warn" q="Is it really the size listed?" a={`≈ ${Math.round(area.from_geometry)} sq ft on this floor`}
            sub={`Listing says ${P1.claim_sqft.toLocaleString()} sq ft. That likely counts other floors, so no verdict.`} />
          <Tile p={pop(f, fps, 105)} kind="none" q="How high are the ceilings?" a="Not on the plan"
            sub="Floor plans never show it. Film a room to measure it." />
        </div>
      </AppFrame>
    </Page>
  );
};

// ---------- 4. furniture fit ----------
const FitScene = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const S = data.synthetic, bed = S.bedroom.polygon;
  const king = S.fits.king, queen = S.fits.queen;   // the app's verdicts (server._best_spot): king "tight", queen "fits"
  const queenOn = f >= 105;
  const kp = pop(f, fps, 28, 14) * (1 - ramp(f, 100, 10));
  const qp = pop(f, fps, 112, 14);
  const W = 1000;
  const kc = centroid(king.placement), qc = centroid(queen.placement);
  const KIND = { fits: "ok", tight: "warn", blocks_path: "bad", no_fit: "bad" };
  const WORD = { fits: "FITS", tight: "TIGHT", blocks_path: "BLOCKS THE WAY OUT", no_fit: "DOESN'T FIT" };
  const kCol = king.verdict === "fits" ? C.ok : king.verdict === "tight" ? C.warn : C.bad;
  const btn = (on) => ({
    padding: "12px 20px", borderRadius: 10, fontSize: 22, fontWeight: 600,
    border: `2px solid ${on ? C.accent : C.line}`, background: on ? C.accentTint : C.panel, color: on ? C.accentInk : C.ink,
  });
  const callout = (kind, chip, text, p) => (
    <div style={{
      opacity: p, transform: `translateY(${(1 - p) * 20}px)`, marginTop: 26, padding: "22px 24px", borderRadius: 14,
      background: { ok: C.okTint, warn: C.warnTint, bad: C.badTint }[kind],
      border: `1px solid ${{ ok: "#bfe9d8", warn: "#f7dfae", bad: "#f6c8c8" }[kind]}`,
    }}>
      <Chip kind={kind}>{chip}</Chip>
      <div style={{ fontSize: 23, marginTop: 14, lineHeight: 1.35 }}>{text}</div>
    </div>
  );
  return (
    <Page style={{ opacity: sceneFade(f, D.fit) }}>
      <AppFrame>
        <PlanView src="synthetic_2br.png" w={S.w} h={S.h} width={W} style={{ left: 50, top: 40 }}>
          <DrawPoly poly={bed} color={C.accent} fill={0.1} sw={4} />
          {kp > 0.01 && (
            <g opacity={kp} transform={`translate(${kc[0]},${kc[1]}) scale(${0.6 + 0.4 * kp}) translate(${-kc[0]},${-kc[1]})`}>
              <polygon points={pts(king.placement)} fill={kCol} fillOpacity="0.3" stroke={kCol} strokeWidth="5" strokeDasharray="16 10" />
              <text x={kc[0]} y={kc[1]} textAnchor="middle" dy="10" fontSize="30" fontWeight="700" fill={C.warnInk}>KING</text>
            </g>
          )}
          {queenOn && (
            <g opacity={qp} transform={`translate(${qc[0]},${qc[1]}) scale(${0.6 + 0.4 * qp}) translate(${-qc[0]},${-qc[1]})`}>
              <polygon points={pts(queen.placement)} fill={C.ok} fillOpacity="0.3" stroke={C.ok} strokeWidth="5" />
              <text x={qc[0]} y={qc[1]} textAnchor="middle" dy="10" fontSize="30" fontWeight="700" fill={C.okInk}>QUEEN</text>
            </g>
          )}
        </PlanView>
        <Caption style={{ left: 54, top: 40 + (W * S.h) / S.w + 8 }}>samples/synthetic_2br.png · {S.bedroom.label}</Caption>
        <div style={{ position: "absolute", left: 1110, top: 60, width: 600 }}>
          <div style={{ fontSize: 40, fontWeight: 700, opacity: ramp(f, 4) }}>Will my bed fit?</div>
          <div style={{ display: "flex", gap: 14, marginTop: 24, opacity: ramp(f, 10) }}>
            <span style={btn(!queenOn)}>King 6'4" × 6'8"</span>
            <span style={btn(queenOn)}>Queen 5' × 6'8"</span>
          </div>
          <div style={{ fontSize: 20, color: C.muted, marginTop: 16, opacity: ramp(f, 14) }}>3 ft walkway on three sides, clear of door swings</div>
          {!queenOn
            ? callout(KIND[king.verdict], WORD[king.verdict],
                `Not with a 3 ft walkway. Tried ${king.orientations_tried} orientations: the closest fit, against a wall, leaves ${king.walkway_ft} ft.`,
                pop(f, fps, 42))
            : callout(KIND[queen.verdict], WORD[queen.verdict],
                "Against the wall with a 3 ft walkway, clear of the door swing. You can still walk to the entrance.", pop(f, fps, 122))}
        </div>
      </AppFrame>
    </Page>
  );
};

// ---------- 5. video scan ----------
const ScanScene = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const H = data.hotel, d = H.dims, c = H.ceiling;
  // each still fades in at a; all but the last fade out at b; the last holds to the end of the scene
  const still = (k, a, b) => (k === 2 ? interpolate(f, [a - 8, a], [0, 1], CLAMP) : interpolate(f, [a - 8, a, b, b + 8], [0, 1, 1, 0], CLAMP));
  const zoom = 1 + f / 900;
  const typed = "show me other layouts".slice(0, Math.max(0, Math.floor((f - 172) / 1.2)));
  const stat = (label, big, sub, start) => {
    const p = pop(f, fps, start);
    return (
      <div style={{ flex: 1, background: C.panel, border: `1px solid ${C.line}`, borderRadius: 14, padding: "16px 18px", opacity: p, transform: `translateY(${(1 - p) * 20}px)` }}>
        <div style={{ fontSize: 17, color: C.muted, fontWeight: 600 }}>{label}</div>
        <div style={{ fontSize: 31, fontWeight: 700, marginTop: 4 }}>{big}</div>
        <div style={{ fontSize: 16, color: C.muted, marginTop: 4 }}>{sub}</div>
      </div>
    );
  };
  return (
    <Page style={{ opacity: sceneFade(f, D.scan) }}>
      <AppFrame right="room scan">
        <div style={{ position: "absolute", left: 40, top: 30, fontSize: 30, fontWeight: 700, opacity: ramp(f, 2) }}>
          At the viewing: film the room for 10 seconds
        </div>
        <div style={{ position: "absolute", left: 40, top: 90, width: 780, height: 439, borderRadius: 26, background: "#0b1220", padding: 14, boxShadow: "0 10px 30px rgba(15,23,42,.25)" }}>
          <div style={{ position: "relative", width: "100%", height: "100%", borderRadius: 14, overflow: "hidden" }}>
            {[0, 1, 2].map((k) => (
              <Img key={k} src={staticFile(`hotel_${k}.jpg`)} style={{
                position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover",
                opacity: still(k, 10 + k * 40, 50 + k * 40), transform: `scale(${zoom})`,
              }} />
            ))}
            <div style={{ position: "absolute", left: 16, top: 14, display: "flex", alignItems: "center", gap: 8, color: "#fff", fontFamily: MONO, fontSize: 18, fontWeight: 700 }}>
              <span style={{ width: 14, height: 14, borderRadius: 7, background: C.bad, opacity: Math.floor(f / 12) % 2 ? 1 : 0.35 }} />
              REC 0:{String(Math.min(9, Math.floor(f / 15))).padStart(2, "0")}
            </div>
          </div>
        </div>
        <div style={{ position: "absolute", left: 40, top: 556, width: 780, display: "flex", gap: 14 }}>
          {stat("Size", `${ftin(d.w.best)} × ${ftin(d.d.best)}`, `likely ${ftin(d.w.low)}–${ftin(d.w.high)} by ${ftin(d.d.low)}–${ftin(d.d.high)}`, 112)}
          {stat("Ceiling", `≈ ${ftin(c.best)}`, `likely ${ftin(c.low)}–${ftin(c.high)}`, 122)}
          {stat("Area", `≈ ${Math.round(H.area_sqft)} sq ft`, `likely ${Math.round(H.area_range[0])}–${Math.round(H.area_range[1])}`, 132)}
        </div>
        <div style={{ position: "absolute", left: 40, top: 700, width: 780, opacity: ramp(f, 160, 10) }}>
          <div style={{ border: `2px solid ${C.accent}`, borderRadius: 12, padding: "14px 18px", fontSize: 24, background: C.panel, display: "flex" }}>
            <span style={{ color: typed ? C.ink : C.muted }}>{typed || "Tell it what to do…"}</span>
            <span style={{ opacity: Math.floor(f / 8) % 2 ? 1 : 0, color: C.accent }}>|</span>
            <span style={{ marginLeft: "auto", background: C.accent, color: "#fff", borderRadius: 8, padding: "2px 14px", fontWeight: 600, fontSize: 20 }}>Do it</span>
          </div>
          <div style={{ display: "flex", gap: 12, marginTop: 16, flexWrap: "wrap" }}>
            {H.options.map((o, i) => {
              const p = pop(f, fps, 204 + i * 8, 14);
              return <span key={o} style={{ background: C.accent, color: "#fff", fontWeight: 600, fontSize: 20, padding: "10px 16px", borderRadius: 10, opacity: p, transform: `scale(${0.7 + 0.3 * p})` }}>{o}</span>;
            })}
          </div>
        </div>
        <div style={{ position: "absolute", left: 880, top: 24, width: 840, height: 830, display: "flex", justifyContent: "center", opacity: ramp(f, 118, 14), transform: `translateX(${(1 - ramp(f, 118, 18)) * 60}px)` }}>
          <Img src={staticFile("cad_hotel_room.png")} style={{ height: "100%", borderRadius: 10, border: `1px solid ${C.line}` }} />
        </div>
      </AppFrame>
    </Page>
  );
};

// ---------- 6. proof ----------
const Bar = ({ label, ours, base, start, f, fps }) => {
  const p = pop(f, fps, start, 26);
  const row = (name, v, col, bold) => (
    <div style={{ display: "flex", alignItems: "center", gap: 18, marginTop: 12 }}>
      <div style={{ width: 230, fontSize: 22, color: bold ? C.ink : C.muted, fontWeight: bold ? 700 : 500 }}>{name}</div>
      <div style={{ height: 36, width: 900 * (v / 100) * p, background: col, borderRadius: 8 }} />
      <div style={{ fontFamily: MONO, fontSize: 26, fontWeight: 700, color: bold ? C.accentInk : C.muted, opacity: p }}>{v.toFixed(v % 1 ? 1 : 0)}%</div>
    </div>
  );
  return (
    <div style={{ marginTop: 40, opacity: ramp(f, start - 4, 10) }}>
      <div style={{ fontSize: 28, fontWeight: 600 }}>{label}</div>
      {row("Footprint", ours, C.accent, true)}
      {row("One Gemini call", base, C.lineStrong, false)}
    </div>
  );
};

const ProofScene = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <Page style={{ opacity: sceneFade(f, D.proof) }}>
      <Grid opacity={0.35} />
      <div style={{ position: "absolute", left: 160, top: 110, width: 1600 }}>
        <div style={{ fontSize: 60, fontWeight: 700, letterSpacing: "-0.02em", opacity: pop(f, fps, 0) }}>Tested on 50 held-out floor plans</div>
        <div style={{ fontSize: 24, color: C.muted, marginTop: 10, opacity: ramp(f, 8) }}>CubiCasa5K test split · full results in eval/results.json</div>
        <Bar f={f} fps={fps} start={20} label="Right when it says VERIFIED" ours={82.0} base={78.6} />
        <Bar f={f} fps={fps} start={50} label="Room count exactly right" ours={38} base={26} />
        <div style={{ marginTop: 46, fontSize: 30, opacity: ramp(f, 90, 12), transform: `translateY(${(1 - ramp(f, 90, 12)) * 16}px)` }}>
          Filmed rooms of known size: the true size was inside the reported range <b style={{ color: C.okInk }}>3 of 3</b> times.
        </div>
      </div>
    </Page>
  );
};

// ---------- 7. end card ----------
const EndScene = () => {
  const f = useCurrentFrame();
  const { fps } = useVideoConfig();
  const a = pop(f, fps, 0), b = ramp(f, 14, 14), c = ramp(f, 30, 14);
  return (
    <Page>
      <Grid opacity={0.55} />
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", opacity: sceneFade(f, D.end, 1, 14) }}>
        <div style={{ display: "flex", alignItems: "center", gap: 24, opacity: a, transform: `scale(${0.9 + 0.1 * a})` }}>
          <Logo size={96} /><span style={{ fontSize: 110, fontWeight: 700, letterSpacing: "-0.03em" }}>Footprint</span>
        </div>
        <div style={{ fontSize: 44, fontWeight: 600, marginTop: 26, opacity: b }}>
          Is that 2BR actually a 2BR? <span style={{ color: C.accent }}>When it can't tell, it says so.</span>
        </div>
        <div style={{ fontSize: 24, color: C.muted, marginTop: 30, opacity: c }}>Gemini 4 Carbon · Neon · Hydrogen · Decisions API · OpenCV</div>
      </AbsoluteFill>
      <Caption style={{ left: 0, right: 0, bottom: 30, textAlign: "center", opacity: c }}>
        Floor plan: "Floor plan 1395 Sq.ft." by SmartVilla, Wikimedia Commons, CC BY-SA 4.0
      </Caption>
    </Page>
  );
};

export const Demo = () => (
  <AbsoluteFill style={{ background: C.bg }}>
    <Series>
      <Series.Sequence durationInFrames={D.hook}><Hook /></Series.Sequence>
      <Series.Sequence durationInFrames={D.plan}><PlanScene /></Series.Sequence>
      <Series.Sequence durationInFrames={D.answers}><AnswersScene /></Series.Sequence>
      <Series.Sequence durationInFrames={D.fit}><FitScene /></Series.Sequence>
      <Series.Sequence durationInFrames={D.scan}><ScanScene /></Series.Sequence>
      <Series.Sequence durationInFrames={D.proof}><ProofScene /></Series.Sequence>
      <Series.Sequence durationInFrames={D.end}><EndScene /></Series.Sequence>
    </Series>
  </AbsoluteFill>
);
