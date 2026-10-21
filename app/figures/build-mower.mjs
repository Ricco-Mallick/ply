// Fig 1 — "Walk it like a mower": the capture tutorial as a reel mower on a lawn.
// The mower only cuts while its wheels turn (walk slowly), and every pass must overlap the last
// (no uncut strip between passes) — which is exactly how a room must be filmed for a clean twin.
//   node app/figures/build-mower.mjs [--light]
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "./kit/iso-kit.mjs";
import * as G from "./kit/lathe.mjs";
import { kitScript } from "./inline-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, "..", "public", "figures");
const theme = process.argv.includes("--light") ? "light" : "dark";

const W = 680, H = 420;
const LAWN = { x: -110, y: -72, w: 220, d: 144, r: 10 };   // 2.2 m x 1.44 m, 1 unit = 1 cm
const STRIPES = 4;                                          // four passes across the lawn
const CUT_W = 48;                                           // the mower's cutting width, cm
const PASS_D = LAWN.d / STRIPES;                            // 36 cm between passes: each one laps the last by 12 cm
const TILES = 24;                                           // cut tiles per pass

// --- the mower, drawn once at x = 0, y = 0 ---------------------------------
const BODY = { x: -22, y: -20, w: 44, d: 40, r: 4 };
const BODY_Z = 9, BODY_H = 13;
const WHEEL_R = 13, WHEEL_W = 2.6, WHEEL_X = 24;
const REEL_R = 8.5, REEL_Z = 12;
const HANDLE_TOP = 41;

// fit to the lawn *plus* the mower at both ends of its travel (handle included), or the grip clips off the frame
const P = k.fitProjection(
  [
    ...k.boxCorners(LAWN, -6, 0),
    [LAWN.x - 54, -30, HANDLE_TOP + 8], [LAWN.x + LAWN.w + 30, 30, HANDLE_TOP + 8],
    [LAWN.x - 54, 30, 0], [LAWN.x + LAWN.w + 30, -30, 0],
  ],
  W, H, { pad: 30, azimuth: 42 },
);
const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (pts, style) => k.dotsSvg(pts, style);
const byDepth = (pts) => [...pts].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));

// --- lawn ------------------------------------------------------------------
function lawn() {
  const out = [`<path class="iso-halo" d="${k.haloOf(LAWN, -6, 6, P)}"/>`, S(k.slabOf(LAWN, -6, 6, P, 10, 1), { tone: "lo" })];
  // soil edge, then the cut tiles: each pass is a row of tiles that flips to "cut" as the mower passes
  out.push(S(k.slabOf(k.insetPlan(LAWN, 2.5), 0, 0.6, P, 8), { tone: "quiet" }));
  for (let row = 0; row < STRIPES; row++) {
    const y = LAWN.y + LAWN.d - PASS_D * (row + 0.5);  // top pass first
    for (let t = 0; t < TILES; t++) {
      const x0 = LAWN.x + (LAWN.w / TILES) * t, w = LAWN.w / TILES - 0.8;
      const tile = { x: x0, y: y - PASS_D / 2 + 1.2, w, d: PASS_D - 2.4, r: 0.6 };
      out.push(`<path class="iso-fill iso-turf" data-pass="${row}" data-tile="${t}" data-edge="${(x0 + w).toFixed(2)}" d="${k.planOutline(tile, 0.7, P, 4)}"/>`);
    }
    // the uncut edge line between passes: the gap the tutorial warns about
    out.push(L(k.lineOnTop([LAWN.x + 3, y - PASS_D / 2], [LAWN.x + LAWN.w - 3, y - PASS_D / 2], 0.9, P), { tone: "faint", dashed: true }));
  }
  for (const [x, y] of k.corners(LAWN, 14)) out.push(S(k.cylinder(x, y, 2.4, 0, 0.8, P, 20), { tone: "mid" }));
  return out.join("");
}

// --- the mower -------------------------------------------------------------
// wheels and the reel are turned on a lathe (their axis is horizontal), the frame and grass box are extrusions,
// and the handle stays are prisms along their own direction.
const wheelF = (y) => G.frameOf([-WHEEL_X, y, BODY_Z + 4], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
const WHEEL_XS = [WHEEL_X, -WHEEL_X];

function wheel(y, w) {
  const F = wheelF(y);
  const disc = S(G.lathe([[0, 0], [0, WHEEL_R], [w, WHEEL_R], [w, 0]], F, P, { steps: 40 }), { tone: "mid" });
  const rim = L(G.circleOf(F, w, WHEEL_R, P, 40), { tone: "lo" });
  const hub = S(G.lathe([[w, 0], [w, 3.4], [w + 1.4, 3.4], [w + 1.4, 0]], F, P, { steps: 20 }), { tone: "hi" });
  const bolts = D(G.dotsOf(F, w + 1.5, 6.4, 6, P, { all: true }), { size: 0.5, tone: "hi" });
  return disc + rim + hub + bolts;
}

function stay(from, to) {
  return L(k.segment(from, to, P), { tone: "lo", className: "iso-handle" });
}

function mower() {
  const out = [`<path class="iso-halo" d="${k.haloOf({ x: -WHEEL_X - 4, y: -BODY.d / 2 - 6, w: (WHEEL_X + 4) * 2, d: BODY.d + 12, r: 8 }, 0.4, 0.6, P)}"/>`];
  // grass box behind the body, with slats
  out.push(S(k.slabOf({ x: -18, y: BODY.d / 2 - 2, w: 36, d: 15, r: 2 }, BODY_Z + 3, BODY_H - 1, P, 4), { tone: "lo" }));
  for (let i = -14; i <= 14; i += 7) out.push(L(k.rise([i, BODY.d / 2 + 5.5], BODY_Z + 4, BODY_Z + BODY_H + 1, P), { tone: "faint" }));
  // the two side plates and the body shell between them
  for (const side of byDepth([[0, -BODY.d / 2 + 3], [0, BODY.d / 2 - 3]])) {
    const y = side[1];
    out.push(S(k.slabOf({ x: -WHEEL_X - 2, y: y - 1.2, w: (WHEEL_X + 2) * 2, d: 2.4, r: 1 }, BODY_Z, BODY_H, P, 4, 0.6), { tone: "mid" }));
    out.push(S(k.cylinder(0, y, 1.2, BODY_Z + BODY_H, 1.2, P, 14), { tone: "hi" }));
  }
  out.push(S(k.slabOf(BODY, BODY_Z, BODY_H, P, 6, 1), { tone: "hi" }));
  out.push(L(k.sideSeam(BODY, BODY_Z + BODY_H, P), { tone: "crease" }));
  // a wide plate over the top with four screws
  out.push(S(k.slabOf({ x: -14, y: -BODY.d / 2 + 4, w: 30, d: BODY.d - 8, r: 2 }, BODY_Z + BODY_H, 1.6, P, 4), { tone: "hi" }));
  for (const [x, y] of [[-11, -14], [13, -14], [-11, 14], [13, 14]]) out.push(S(k.cylinder(x, y, 1.1, BODY_Z + BODY_H + 1.6, 0.7, P, 16), { tone: "lo" }));
  // the reel: turned drum with five blades cut across its face
  const reelF = G.frameOf([0, -BODY.d / 2 + 6, REEL_Z], [0, 1, 0], [1, 0, 0], [0, 0, 1]);
  out.push(S(G.lathe([[0, 0], [0, REEL_R], [BODY.d - 12, REEL_R], [BODY.d - 12, 0]], reelF, P, { steps: 32 }), { tone: "lo" }));
  for (let i = 0; i < 5; i++) {
    const a = (i / 5) * Math.PI * 2;
    out.push(L(k.pathOf(k.pointsAt([[Math.cos(a) * REEL_R, -BODY.d / 2 + 6, REEL_Z + Math.sin(a) * REEL_R], [Math.cos(a) * REEL_R, BODY.d / 2 - 6, REEL_Z + Math.sin(a) * REEL_R]].map(([x, y, z]) => [x, y, z]), P)), { tone: "hi" }));
  }
  // wheels: one each side, outside the plates
  out.push(...byDepth([-BODY.d / 2 + 2, BODY.d / 2 - 2]).map((y) => wheel(y - 2, WHEEL_W)));  // y is a number here
  // cutting bar in front with three bolts
  out.push(S(k.slabOf({ x: -WHEEL_X + 2, y: -BODY.d / 2 - 1, w: WHEEL_X * 2 - 4, d: 2.6, r: 0.6 }, 3, 3.2, P, 3), { tone: "hi" }));
  for (const x of [-12, 0, 12]) out.push(S(k.cylinder(x, -BODY.d / 2, 1.0, 6.2, 0.7, P, 14), { tone: "lo" }));
  // handle: two stays to a grip, with plates at the frame
  const gripZ = HANDLE_TOP, gripX = -WHEEL_X - 24;
  for (const yy of byDepth([-BODY.d / 2 + 7, BODY.d / 2 - 7])) {
    out.push(stay([-WHEEL_X + 3, yy, BODY_Z + BODY_H + 2], [gripX + 2, yy, gripZ]));
    out.push(S(k.cylinder(-WHEEL_X + 3, yy, 1.3, BODY_Z + BODY_H + 2, 1.2, P, 14), { tone: "hi" }));
    out.push(S(k.slabOf({ x: -WHEEL_X + 1, y: yy - 2.4, w: 5, d: 4.8, r: 0.8 }, BODY_Z + BODY_H, 2, P, 3), { tone: "mid" }));
  }
  out.push(L(k.segment([gripX + 2, -BODY.d / 2 + 7, gripZ], [gripX + 2, BODY.d / 2 - 7, gripZ], P), { tone: "hi", className: "iso-handle" }));
  // a small measuring wheel on the side: the distance walked
  out.push(wheel(BODY.d / 2 + 5.4, 1.6));
  return out.join("");
}

const O = at([0, 0, 0]);
const ART_S = 1.25;  // a touch larger than life so the parts read; the lawn keeps its true 2.2 m
const artTransform = `translate(${O[0].toFixed(2)} ${O[1].toFixed(2)}) scale(${ART_S}) translate(${(-O[0]).toFixed(2)} ${(-O[1]).toFixed(2)})`;
const svg = `<svg class="iso-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="A reel mower on a lawn, drawn in isometric, cutting a stripe as it is pushed along one pass.">
  <g id="lawn">${lawn()}</g>
  <g id="mower"><g id="mower-art" transform="${artTransform}">${mower()}</g></g>
</svg>`;

// --- behaviour -------------------------------------------------------------
const PASS_Y = Array.from({ length: STRIPES }, (_, i) => LAWN.y + LAWN.d - PASS_D * (i + 0.5));
const live = `
const P = ${JSON.stringify(P)};
const LAWN = ${JSON.stringify(LAWN)};
const PASS_Y = ${JSON.stringify(PASS_Y)};
const PASS_D = ${PASS_D.toFixed(3)};
const CUT_W = ${CUT_W};
const X0 = ${(LAWN.x + 44).toFixed(2)}, X1 = ${(LAWN.x + LAWN.w - 44).toFixed(2)};
const mower = document.getElementById("mower");
const plate = document.querySelector(".iso-plate");
const readout = document.querySelector('[data-corner="readout"]');
const tiles = [...document.querySelectorAll(".iso-turf")];
let pass = 0, x = X0, target = x, touring = true, clock = 0, frame = 0, last = 0, visible = false, idle = 0;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const cutTo = (px) => { for (const t of tiles) if (+t.dataset.pass <= pass && +t.dataset.edge <= px + 0.001) t.classList.add("cut"); };
function draw() {
  mower.setAttribute("transform", translateAlong(P, [x, PASS_Y[pass], 0]));
  const cut = Math.max(0, x - X0);
  cutTo(x);
  const text = "pass " + (pass + 1) + " of " + PASS_Y.length + " · " + (cut / 100).toFixed(2) + " m cut · " + (CUT_W - PASS_D).toFixed(0) + " cm overlap";
  if (readout && readout.textContent !== text) readout.textContent = text;
}
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60; last = now;
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (touring) {
    clock += dt;
    const lap = clock % 9;
    if (lap < 4.4) target = X0 + (X1 - X0) * (lap / 4.4);
    else if (lap < 5.2) target = X1;
    else { target = X0; if (lap > 8.6) { pass = (pass + 1) % PASS_Y.length; } }
  }
  x = calm ? target : x + (target - x) * (1 - Math.exp(-dt / 0.18));
  draw();
  frame = visible ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
const steer = (next, nextPass) => {
  touring = false; if (nextPass !== undefined) pass = nextPass; target = clamp(next, X0, X1);
  clearTimeout(idle); idle = setTimeout(() => { touring = true; clock = 0; run(); }, 4200); run();
};
new IntersectionObserver((e) => { visible = e[e.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px" }).observe(document.querySelector(".iso-stage"));
document.querySelector(".iso-stage").addEventListener("pointermove", (ev) => {
  const box = document.querySelector(".iso-svg").getBoundingClientRect();
  const share = clamp((ev.clientX - box.left) / box.width, 0, 1);
  const row = clamp(Math.floor((ev.clientY - box.top) / box.height * PASS_Y.length), 0, PASS_Y.length - 1);
  steer(X0 + share * (X1 - X0), PASS_Y.length - 1 - row);
});
document.querySelector(".iso-stage").addEventListener("pointerleave", () => steer(target));
document.querySelector(".iso-stage").addEventListener("keydown", (ev) => {
  if (ev.key === "ArrowRight") { ev.preventDefault(); steer(target + 18); }
  if (ev.key === "ArrowLeft") { ev.preventDefault(); steer(target - 18); }
  if (ev.key === "ArrowUp") { ev.preventDefault(); pass = Math.max(0, pass - 1); run(); }
  if (ev.key === "ArrowDown") { ev.preventDefault(); pass = Math.min(PASS_Y.length - 1, pass + 1); run(); }
});
draw();
`;

const body = k.plateHtml({
  fig: "Fig 1",
  title: "Walk it like a mower",
  hint: "Move along the lawn to push a pass",
  readout: "pass 1 of 6 · 0.00 m cut · 32 cm overlap",
  keys: [
    { mark: "lit", label: "The reel: it only cuts while the wheels turn, so steady beats quick" },
    { mark: "raised", label: "The overlap: each pass must lap the last, or a strip is left uncut" },
  ],
  caption:
    "A reel mower on a lawn. The reel cuts only while the wheels turn, so speed buys nothing and a steady push cuts clean; and each pass has to overlap the one before it, or a strip is left standing. Filming a room works the same way: slow, level, and overlapping — that is what leaves the twin without holes.",
  body: `<style>
  .iso { --anatomy-page:#070b14; --anatomy-card:#0b1020; --anatomy-paper:#0e1524; --anatomy-face:#16203a; --anatomy-top:#1b2743;
         --anatomy-shade-0:#0a1120; --anatomy-shade-1:#0d1526; --anatomy-shade-2:#101a2e; --anatomy-shade-3:#132038;
         --anatomy-lit-top:#2b3f6d; --anatomy-lit-shade:#1e2c4e; --anatomy-hi:#e2e9fb; --anatomy-mid:#93a6cd;
         --anatomy-lo:#5d6f95; --anatomy-faint:#354463; --anatomy-lit:#a8d4ff; --anatomy-halo-opacity:.55; }
  .iso-turf { fill: #0d1830; stroke: #17263f; stroke-width: .4; }
  .iso-turf.cut { fill: #26476f; stroke: #3f6ba3; stroke-width: .5; }
  .iso-svg { max-height: 540px; }
  .iso-grip { stroke: var(--anatomy-lit); }
  .iso-handle { stroke: var(--anatomy-lo); stroke-width: 2.8px; stroke-linecap: round; }
</style><div class="iso-stage" tabindex="0" role="application" aria-label="Reel mower on a lawn: move the pointer along the lawn to push a pass">${svg}</div>`,
});

mkdirSync(OUT, { recursive: true });
const file = join(OUT, theme === "light" ? "mower-light.html" : "mower.html");
writeFileSync(file, k.pageHtml({ title: "Walk it like a mower", theme, body, script: kitScript() + live }));
console.log("wrote", file);
