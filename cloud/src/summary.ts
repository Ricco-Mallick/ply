// Copyright (c) 2026 Ricco Marcelino. All rights reserved. Proprietary and confidential — see LICENSE.
/* A twin described for machines: GET /v1/scenes/:id/summary (JSON) and /summary.txt (plain text for a chatbot or an
   agent that reads pages without running them). Built from the files the twin already publishes (scene.json,
   occupancy.json, shell.json, places.json): no GPU, nothing recomputed, cheap enough for the free Workers tier.
   Walking routes stay in the viewer; every place carries the link that opens one. */

export interface SummaryInput {
  id: string; site: string; api: string;
  scene: Record<string, any> | null; occupancy: Record<string, any> | null; shell: Record<string, any> | null;
  places: Record<string, any> | null;
}

const r2 = (v: number) => Math.round(v * 100) / 100;
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

export function summarize(i: SummaryInput) {
  const s = i.scene ?? {}, b = s.bounds, occ = i.occupancy ?? {}, sh = i.shell ?? {};
  const size = b?.min && b?.max ? { x: r2(b.max[0] - b.min[0]), y: r2(b.max[1] - b.min[1]), z: r2(b.max[2] - b.min[2]) } : null;
  const cell = finite(occ.cellSize) ? occ.cellSize : finite(occ.cell) ? occ.cell : null;
  const freeCells = finite(occ.counts?.free) ? occ.counts.free : finite(occ.free) ? occ.free : null;
  const walkable = finite(occ.walk?.standableM2) ? occ.walk.standableM2 : cell && freeCells ? freeCells * cell * cell : null;
  const walls = Array.isArray(sh.walls) ? sh.walls : [];
  const sceneUrl = `${i.api}/v1/scenes/${i.id}/scene.json`, walk = `${i.site}/walk.html?scene=${encodeURIComponent(sceneUrl)}`;
  const list = (Array.isArray(i.places?.places) ? i.places!.places : []).filter((p: any) => p && typeof p.name === 'string' && Array.isArray(p.p) && p.p.every(finite));
  const places = list.slice(0, 200).map((p: any) => ({
    id: p.id, name: p.name, category: p.cat, position: p.p.map(r2), heightAboveFloorM: r2(p.p[1] - (s.floorY ?? 0)),
    ...(p.info?.note ? { note: String(p.info.note).slice(0, 600) } : {}), ...(p.info?.link ? { link: p.info.link } : {}),
    ...(p.ai ? { detectedAs: p.ai.cls, confidence: r2(p.ai.conf) } : {}),
    open: `${walk}&place=${encodeURIComponent(p.id)}`,
  }));
  // straight-line distances between the pinned places (or the first ten), so "how far is X from Y" needs no maths
  const key = places.filter((_: unknown, k: number) => list[k].pinned).slice(0, 10);
  const near = (key.length >= 2 ? key : places.slice(0, 10));
  const distances = near.flatMap((a: any, k: number) => near.slice(k + 1).map((c: any) => ({
    from: a.name, to: c.name, straightM: r2(Math.hypot(a.position[0] - c.position[0], a.position[2] - c.position[2])) })));
  return {
    schema: 'ply.twin-summary/1', id: i.id,
    frame: 'metres; +y up; floor at y = ' + (s.floorY ?? 0) + '; x and z across the floor',
    scale: { source: s.scaleSource ?? null, note: s.scaleSource === 'ar-poses' ? 'measured by the phone (ARKit / ARCore tracking)' : s.scaleSource === 'camera-height-prior' ? 'estimated from the camera height' : null },
    sizeM: size, ceilingM: finite(sh.ceilingY) ? r2(sh.ceilingY - (s.floorY ?? 0)) : null,
    walkableM2: walkable === null ? null : r2(walkable), walls: walls.length,
    longestWallM: walls.length ? r2(Math.max(...walls.map((w: any) => finite(w.length) ? w.length : 0))) : null,
    places, distances,
    links: { walkthrough: walk, sceneJson: sceneUrl, placesJson: `${i.api}/v1/scenes/${i.id}/places` },
    credit: 'Captured and reconstructed with Ply (ricco.codes/ply). Positions are in the twin\'s own frame, not on a world map.',
  };
}

export function summaryText(j: ReturnType<typeof summarize>): string {
  const L: string[] = [];
  L.push(`# Ply twin ${j.id}`, '');
  if (j.sizeM) L.push(`Size: ${j.sizeM.x} m by ${j.sizeM.z} m across the floor, ${j.sizeM.y} m tall overall.`);
  if (j.ceilingM !== null) L.push(`Ceiling: ${j.ceilingM} m above the floor.`);
  if (j.walkableM2 !== null) L.push(`Walkable floor: about ${Math.round(j.walkableM2)} m².`);
  if (j.walls) L.push(`Walls found: ${j.walls}${j.longestWallM ? `, the longest ${j.longestWallM} m` : ''}.`);
  if (j.scale.note) L.push(`Scale: ${j.scale.note}.`);
  L.push(`Coordinates: ${j.frame}.`, '');
  L.push(j.places.length ? `## Places (${j.places.length})` : '## Places', '');
  if (!j.places.length) L.push('None tagged yet.');
  for (const p of j.places) {
    L.push(`- ${p.name} (${p.category}) at x ${p.position[0]}, z ${p.position[2]}, ${p.heightAboveFloorM} m up${p.detectedAs ? `; detected as ${p.detectedAs}` : ''}`);
    if (p.note) L.push(`  Note: ${p.note.replace(/\s+/g, ' ')}`);
    if (p.link) L.push(`  Link: ${p.link}`);
    L.push(`  Open in 3D: ${p.open}`);
  }
  if (j.distances.length) { L.push('', '## Straight-line distances', ''); for (const d of j.distances) L.push(`- ${d.from} to ${d.to}: ${d.straightM} m`); }
  L.push('', `Walk it: ${j.links.walkthrough}`, `Data: ${j.links.sceneJson} and ${j.links.placesJson}`, '', j.credit, '');
  return L.join('\n');
}
