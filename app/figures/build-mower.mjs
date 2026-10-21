// Fig 1 — "Walk it like a mower": the capture tutorial as a reel mower on a lawn.
//
// The object is the explanation: a reel mower cuts only while its wheels turn (walk steadily, don't rush) and each
// pass must lap the last (overlap the sweep) or a strip is left standing — exactly the rule for filming a room so
// the twin comes back without holes. Every number in the readout comes from the same geometry that moves the parts.
//
//   node app/figures/build-mower.mjs [--light]
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "./kit/iso-kit.mjs";
import * as G from "./kit/lathe.mjs";
import { tubePieces, tubeSvg } from "./kit/tube.mjs";
import { kitScript } from "./inline-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, "..", "public", "figures");
const theme = process.argv.includes("--light") ? "light" : "dark";

const W = 660, H = 430;
const LAWN = { x: -116, y: -78, w: 232, d: 156, r: 9 };       // 2.32 m x 1.56 m, 1 unit = 1 cm
const STRIPES = 4;
const PASS_D = LAWN.d / STRIPES;                              // 39 cm between passes
const TILES = 26;
const CUT_W = 52;                                             // cutting width, cm
const OVERLAP = CUT_W - PASS_D;                               // 13 cm: each pass laps the last

const BODY = { x: -26, y: -21, w: 52, d: 42, r: 4 };
const BODY_Z = 10, BODY_H = 15;
const WHEEL_R = 14, WHEEL_W = 2.4, WHEEL_X = 27;
const REEL_R = 8.6, REEL_Z = 13, REEL_Y = BODY.d - 16;
const BAR_Z = 4;
const HANDLE_TOP = 44;
const GRIP_X = -WHEEL_X - 15;

const P = k.fitProjection(
  [
    ...k.boxCorners(LAWN, -7, 0),
    [LAWN.x - 30, -34, HANDLE_TOP + 10], [LAWN.x + LAWN.w + 30, 34, HANDLE_TOP + 10],
    [LAWN.x - 30, 34, 0], [LAWN.x + LAWN.w + 30, -34, 0],
  ],
  W, H, { pad: 26, azimuth: 50 },
);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (pts, style) => k.dotsSvg(pts, style);
const at = (p) => k.iso(p, P);
const byDepth = (...pts) => pts.map((p) => [k.depthOf(p, P), p]).sort((a, b) => a[0] - b[0]).map(([, p]) => p);

// ---------------------------------------------------------------- the lawn
function lawn() {
  const out = [`<path class="iso-halo" d="${k.haloOf(LAWN, -7, 7, P)}"/>`, S(k.slabOf(LAWN, -7, 7, P, 10, 1.2), { tone: "lo" })];
  out.push(S(k.slabOf(k.insetPlan(LAWN, 3), 0, 0.7, P, 8), { tone: "quiet" }));
  // the mown surface: tiles that flip to "cut" behind the mower
  for (let row = 0; row < STRIPES; row++) {
    const y = LAWN.y + LAWN.d - PASS_D * (row + 0.5);
    for (let t = 0; t < TILES; t++) {
      const x0 = LAWN.x + (LAWN.w / TILES) * t, w = LAWN.w / TILES - 0.7;
      const tile = { x: x0, y: y - PASS_D / 2 + 1.4, w, d: PASS_D - 2.8, r: 0.5 };
      out.push(`<path class="iso-fill iso-turf" data-pass="${row}" data-tile="${t}" data-edge="${(x0 + w).toFixed(2)}" d="${k.planOutline(tile, 0.8, P, 4)}"/>`);
    }
    // the seam between two passes: the line the overlap has to cross
    out.push(L(k.lineOnTop([LAWN.x + 4, y - PASS_D / 2], [LAWN.x + LAWN.w - 4, y - PASS_D / 2], 1.0, P), { tone: "faint", dashed: true }));
  }
  // corner stakes and a ruler along the near edge
  for (const [x, y] of k.corners(LAWN, 11)) {
    out.push(S(k.cylinder(x, y, 2.6, 0, 1.0, P, 20), { tone: "mid" }));
    out.push(S(k.cylinder(x, y, 1.2, 1.0, 0.5, P, 12), { tone: "hi" }));
  }
  for (let i = 1; i < 5; i++) {
    const x = LAWN.x + (LAWN.w / 5) * i;
    out.push(L(k.segment([x, LAWN.y + LAWN.d - 2.4, 0.9], [x, LAWN.y + LAWN.d - 6.4, 0.9], P), { tone: "faint" }));
  }
  return out.join("");
}

// ---------------------------------------------------------------- the mower
function wheel(y, w) {
  const F = G.frameOf([-WHEEL_X, y, BODY_Z + 4], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
  const out = [];
  out.push(S(G.lathe([[0, 0], [0, WHEEL_R], [w, WHEEL_R], [w, 0]], F, P, { steps: 44 }), { tone: "mid" }));
  out.push(L(G.circleOf(F, w, WHEEL_R, P, 44), { tone: "lo" }));
  out.push(L(G.circleOf(F, w, WHEEL_R - 2.1, P, 44), { tone: "faint" }));
  out.push(D(G.dotsOf(F, w, WHEEL_R - 1.0, 14, P, { all: true }), { size: 0.4, tone: "lo" }));
  out.push(S(G.lathe([[w, 0], [w, 3.6], [w + 1.5, 3.6], [w + 1.5, 2.2], [w + 0.4, 1.4], [w + 0.4, 0]], F, P, { steps: 26 }), { tone: "hi" }));
  out.push(D(G.dotsOf(F, w + 1.6, 6.6, 6, P, { all: true, phase: Math.PI / 6 }), { size: 0.5, tone: "hi" }));
  return out.join("");
}

function reel() {
  const out = [];
  const y0 = -REEL_Y / 2;
  const F = G.frameOf([0, y0, REEL_Z], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
  out.push(S(G.lathe([[0, 0], [0, REEL_R], [REEL_Y, REEL_R], [REEL_Y, 0]], F, P, { steps: 34 }), { tone: "lo" }));
  // five blades: plates standing off the drum face, each on its own frame turned about the drum axis
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2 + 0.3;
    const BF = G.frameOf([0, y0 + 1.5, REEL_Z], [0, 1, 0], [Math.cos(a), 0, Math.sin(a)], [-Math.sin(a), 0, Math.cos(a)]);
    out.push(S(G.prismOf(BF, [[REEL_R - 1.6, -0.35], [REEL_R + 0.7, -0.15], [REEL_R + 0.7, 0.15], [REEL_R - 1.6, 0.35]], 0, REEL_Y - 3, P), { tone: "hi" }));
  }
  for (const yy of [y0 + 0.8, y0 + REEL_Y - 2.2]) out.push(S(k.cylinder(0, yy, 1.5, 0, 0.6, P, 14), { tone: "mid" }));
  return out.join("");
}

function body() {
  const out = [];
  // grass box behind the body, slatted, with a hinge and two bolts
  const BOX = { x: -19, y: BODY.d / 2 - 1, w: 38, d: 15, r: 2 };
  out.push(S(k.slabOf(BOX, BODY_Z + 4, BODY_H - 1, P, 4, 0.6), { tone: "lo" }));
  for (let i = -13; i <= 13; i += 6.5) out.push(L(k.rise([i, BOX.y + BOX.d - 0.6], BODY_Z + 4.4, BODY_Z + BODY_H + 1.4, P), { tone: "faint" }));
  out.push(L(k.sideSeam(BOX, BODY_Z + BODY_H + 3, P), { tone: "crease" }));
  for (const x of [-16, 16]) out.push(S(k.cylinder(x, BOX.y + 1.4, 1.2, BODY_Z + BODY_H + 3, 0.8, P, 14), { tone: "hi" }));
  // the side plates, back one first, each with four bolt heads
  for (const p of byDepth([0, -BODY.d / 2 + 3, 0], [0, BODY.d / 2 - 3, 0])) {
    const yy = p[1];
    out.push(S(k.slabOf({ x: -WHEEL_X - 2, y: yy - 1.2, w: (WHEEL_X + 2) * 2, d: 2.4, r: 1 }, BODY_Z, BODY_H, P, 4, 0.7), { tone: "mid" }));
    for (const x of [-WHEEL_X + 1, WHEEL_X - 1]) for (const z of [BODY_Z + 2, BODY_Z + BODY_H - 3]) out.push(S(k.cylinder(x, yy, 1.4, z, 0.7, P, 14), { tone: "hi" }));
  }
  // the shell between the plates, its top plate, four screws and a badge
  out.push(S(k.slabOf(BODY, BODY_Z, BODY_H, P, 6, 1), { tone: "hi" }));
  out.push(L(k.sideSeam(BODY, BODY_Z + BODY_H, P), { tone: "crease" }));
  out.push(S(k.slabOf({ x: -17, y: -BODY.d / 2 + 5, w: 34, d: BODY.d - 10, r: 2 }, BODY_Z + BODY_H, 1.8, P, 4), { tone: "hi" }));
  for (const [x, y] of [[-13, -14], [13, -14], [-13, 14], [13, 14]]) out.push(S(k.cylinder(x, y, 1.1, BODY_Z + BODY_H + 1.8, 0.7, P, 16), { tone: "lo" }));
  out.push(L(k.lineOnTop([-6, BODY.y + BODY.d - 0.1], [8, BODY.y + BODY.d - 0.1], BODY_Z + 7, P), { tone: "hi" }));
  // cutting bar in front, with three bolts
  out.push(S(k.slabOf({ x: -WHEEL_X + 3, y: -BODY.d / 2 - 1.5, w: WHEEL_X * 2 - 6, d: 2.8, r: 0.6 }, BAR_Z, 3.0, P, 3), { tone: "hi" }));
  for (const x of [-13, 0, 13]) out.push(S(k.cylinder(x, -BODY.d / 2, 1.0, BAR_Z + 3.0, 0.7, P, 14), { tone: "lo" }));
  return out.join("");
}

function handle() {
  const out = [];
  for (const p of byDepth([0, -BODY.d / 2 + 7, 0], [0, BODY.d / 2 - 7, 0])) {
    const y = p[1];
    const route = [[-WHEEL_X + 4, y, BODY_Z + BODY_H + 1], [-WHEEL_X - 6, y, 30], [GRIP_X + 2, y, HANDLE_TOP]];
    for (const part of tubePieces(route, 0.95, P, { maxLength: 400, spacing: 2 })) out.push(tubeSvg(part, { tone: "mid" }));
    out.push(S(k.slabOf({ x: -WHEEL_X + 1.5, y: y - 2.2, w: 5.5, d: 4.4, r: 0.8 }, BODY_Z + BODY_H, 1.6, P, 3), { tone: "hi" }));
    out.push(S(k.cylinder(-WHEEL_X + 4, y, 1.1, BODY_Z + BODY_H + 1.6, 0.6, P, 14), { tone: "lo" }));
  }
  const GF = G.frameOf([GRIP_X + 2, -BODY.d / 2 + 7, HANDLE_TOP], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
  out.push(S(G.lathe([[0, 0], [0, 1.9], [BODY.d - 14, 1.9], [BODY.d - 14, 0]], GF, P, { steps: 22 }), { tone: "lo", className: "iso-grip" }));
  for (const s of [0.6, BODY.d - 14.6]) out.push(L(G.circleOf(GF, s, 1.9, P, 22), { tone: "faint" }));
  return out.join("");
}

function gauge() {
  const out = [];
  const y = BODY.d / 2 + 6.5;
  out.push(S(k.slabOf({ x: -9, y: y - 1, w: 12, d: 2, r: 0.6 }, BODY_Z + BODY_H + 2, 1.4, P, 3), { tone: "mid" }));
  const F = G.frameOf([-3, y, BODY_Z + BODY_H + 3], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
  out.push(S(G.lathe([[0, 0], [0, 3.4], [1.2, 3.4], [1.2, 0]], F, P, { steps: 24 }), { tone: "hi" }));
  out.push(L(G.circleOf(F, 1.2, 2.6, P, 24), { tone: "lo" }));
  out.push(L(G.circleOf(F, 1.25, 1.1, P, 16), { tone: "hi" }));
  return out.join("");
}

const mower = () => [body(), reel(), wheel(-BODY.d / 2 + 2 - WHEEL_W, WHEEL_W), wheel(BODY.d / 2 - 2, WHEEL_W), handle(), gauge()].join("");
const O = at([0, 0, 0]);
const ART_S = 1.18;   // a touch larger than life so the parts read; the lawn keeps its true 2.32 m
const artTransform = `translate(${O[0].toFixed(2)} ${O[1].toFixed(2)}) scale(${ART_S}) translate(${(-O[0]).toFixed(2)} ${(-O[1]).toFixed(2)})`;

const svg = `<svg class="iso-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="A reel mower on a lawn, drawn in isometric: pushing it along a pass cuts that stripe, and each pass overlaps the last.">
  <g id="lawn">${lawn()}</g>
  <g id="mower"><g transform="${artTransform}">${mower()}</g></g>
</svg>`;

// ---------------------------------------------------------------- behaviour
const PASS_Y = Array.from({ length: STRIPES }, (_, i) => LAWN.y + LAWN.d - PASS_D * (i + 0.5));
const live = `
const P = ${JSON.stringify(P)};
const PASS_Y = ${JSON.stringify(PASS_Y)};
const PASS_D = ${PASS_D.toFixed(3)}, CUT_W = ${CUT_W};
const X0 = ${(LAWN.x + 62).toFixed(2)}, X1 = ${(LAWN.x + LAWN.w - 62).toFixed(2)};
const mower = document.getElementById("mower");
const readout = document.querySelector('[data-corner="readout"]');
const tiles = [...document.querySelectorAll(".iso-turf")];
let pass = 0, x = X0, target = x, touring = true, clock = 0, frame = 0, last = 0, visible = false, idle = 0;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
function draw() {
  mower.setAttribute("transform", translateAlong(P, [x, PASS_Y[pass], 0]));
  for (const t of tiles) if (+t.dataset.pass <= pass && +t.dataset.edge <= x + 0.001) t.classList.add("cut");
  const cut = Math.max(0, x - X0);
  const text = "pass " + (pass + 1) + " of " + PASS_Y.length + " · " + (cut / 100).toFixed(2) + " m cut · " + (CUT_W - PASS_D).toFixed(0) + " cm overlap";
  if (readout && readout.textContent !== text) readout.textContent = text;
}
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60; last = now;
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (touring) {
    clock += dt;
    const lap = clock % 10;
    if (lap < 5) target = X0 + (X1 - X0) * (lap / 5);
    else if (lap < 6.2) target = X1;
    else { target = X0; if (lap > 9.6) pass = (pass + 1) % PASS_Y.length; }
  }
  x = calm ? target : x + (target - x) * (1 - Math.exp(-dt / 0.16));
  draw();
  frame = visible ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
const steer = (next, nextPass) => {
  touring = false; if (nextPass !== undefined) pass = nextPass; target = clamp(next, X0, X1);
  clearTimeout(idle); idle = setTimeout(() => { touring = true; clock = 0; run(); }, 4200); run();
};
new IntersectionObserver((e) => { visible = e[e.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px" }).observe(document.querySelector(".iso-stage"));
const stage = document.querySelector(".iso-stage"), svgEl = document.querySelector(".iso-svg");
stage.addEventListener("pointermove", (ev) => {
  const box = svgEl.getBoundingClientRect();
  const share = clamp((ev.clientX - box.left) / box.width, 0, 1);
  const row = clamp(Math.floor((((ev.clientY - box.top) / box.height) - 0.18) / 0.62 * PASS_Y.length), 0, PASS_Y.length - 1);
  steer(X0 + share * (X1 - X0), PASS_Y.length - 1 - row);
});
stage.addEventListener("pointerleave", () => steer(target));
stage.addEventListener("keydown", (ev) => {
  if (ev.key === "ArrowRight") { ev.preventDefault(); steer(target + 16); }
  if (ev.key === "ArrowLeft") { ev.preventDefault(); steer(target - 16); }
  if (ev.key === "ArrowUp") { ev.preventDefault(); pass = Math.max(0, pass - 1); run(); }
  if (ev.key === "ArrowDown") { ev.preventDefault(); pass = Math.min(PASS_Y.length - 1, pass + 1); run(); }
});
draw();
`;

const bodyHtml = k.plateHtml({
  fig: "Fig 1",
  title: "Walk it like a mower",
  hint: "Move along the lawn to push a pass",
  readout: `pass 1 of ${STRIPES} · 0.00 m cut · ${OVERLAP.toFixed(0)} cm overlap`,
  keys: [
    { mark: "lit", label: "The reel: five blades on a drum, cutting only while the wheels turn" },
    { mark: "raised", label: "The overlap: each pass laps the last, so no strip is left standing" },
  ],
  caption:
    "A reel mower on a lawn. The reel cuts only while the wheels turn, so a steady push cuts clean and speed buys nothing; and each pass has to overlap the one before it, or a strip is left standing. Filming a room is the same rule: slow, level, and overlapping — that is what leaves the twin without holes.",
  body: `<style>
  .iso { --anatomy-page:#070b14; --anatomy-card:#0b1020; --anatomy-paper:#0e1524; --anatomy-face:#182343; --anatomy-top:#1e2b50;
         --anatomy-shade-0:#0a1120; --anatomy-shade-1:#0d1526; --anatomy-shade-2:#101a2e; --anatomy-shade-3:#142039;
         --anatomy-lit-top:#31477a; --anatomy-lit-shade:#213156; --anatomy-hi:#e6edff; --anatomy-mid:#9db0d8;
         --anatomy-lo:#63779e; --anatomy-faint:#374766; --anatomy-lit:#b7dcff; --anatomy-halo-opacity:.6; }
  .iso-turf { fill: #0e1a31; stroke: none; }
  .iso-turf[data-pass="1"], .iso-turf[data-pass="3"] { fill: #101d37; }
  .iso-turf.cut { fill: #2d5387; }
  .iso-turf.cut[data-pass="1"], .iso-turf.cut[data-pass="3"] { fill: #31598f; }
  .iso-grip { fill: var(--anatomy-lo); }
  .iso-svg { max-height: 520px; }
</style><div class="iso-stage" tabindex="0" role="application" aria-label="Reel mower on a lawn: move the pointer along the lawn to push a pass">${svg}</div>`,
});

mkdirSync(OUT, { recursive: true });
const file = join(OUT, theme === "light" ? "mower-light.html" : "mower.html");
writeFileSync(file, k.pageHtml({ title: "Walk it like a mower", theme, body: bodyHtml, script: kitScript() + live }));
console.log("wrote", file);
