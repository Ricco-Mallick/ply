# Places API (round 2 contract)

The walkthrough's pins (places) for a scene are readable and writable over HTTP. The app should stop fetching
`places.json` next to the scene and use these two routes instead: the file is still the store, but this contract
guarantees fresh reads and validated writes.

A scene id is the job id. Places live at `scenes/<sceneId>/places.json`, next to `scene.json`.

## GET /v1/scenes/:id/places

Public. No key, no publish gate (a scene in review can be tagged before it is approved).

`200` with exactly three fields:

```json
{
  "schema": 1,
  "updatedAt": "2026-11-20T10:11:12.000Z",
  "places": [
    { "id": "main-door", "name": "Main door", "cat": "Exits", "p": [-0.98, 1.2, 2.19], "pinned": true,
      "ai": { "cls": "door", "conf": 0.91, "half": [0.45, 1.0, 0.08] } }
  ]
}
```

- No file yet: `200 {"schema": 1, "updatedAt": null, "places": []}`. There is no 404 for a scene without pins.
- `updatedAt` is `null` for files the pipeline wrote before this route existed (they carry `version` and `source`
  instead of `schema` and `updatedAt`; both shapes read as `schema: 1`).
- A file that is not a places document answers `502 {"error": "places.json is not readable"}`.
- `Cache-Control: no-store` on every answer.

`p` is metres with the floor at `y = 0`. `ai` is the AI detection box: class, confidence, half-size. `pinned` marks
the pins that show on the map ribbon.

## PUT /v1/scenes/:id/places

Owner only. Body is the **whole list** (`{"places": [...]}`): it replaces the file, so send everything, not a diff.
Concurrent writers are last-write-wins; there is no ETag or `If-Match` yet.

| Credential | Result |
|---|---|
| `X-Ply-Key: <APP_KEY>` (the team key) | allowed |
| `Authorization: Bearer <WORKER_TOKEN>` (the GPU worker) | allowed |
| A valid invite code, or no key, or a wrong key | `401` for no or wrong key, `403` for an invite |

Invites let a guest spend GPU time on a job. They cannot rewrite a scene's pins.

Success: `200`

```json
{ "ok": true, "updatedAt": "2026-11-20T10:11:12.000Z", "count": 2 }
```

`updatedAt` in the response is the timestamp stored in the file.

Errors name the place index and the field that failed:

```json
{ "ok": false, "error": "place 3: cat must be one of Furniture, Storage, Exits, Things (got \"Doors\")",
  "index": 3, "field": "cat" }
```

`index` is `null` for a body-level problem. `field` is `places` or `body` for those, and one of `id`, `name`, `cat`,
`p`, `ai`, `ai.cls`, `ai.conf`, `ai.half`, `pinned`, `place` for a place.

Statuses: `400` validation, `401` missing or wrong key, `403` invite, `405` method (with `Allow`), `413` body over
512 KB, `502` storage refused the write.

### Validation rules

| Field | Rule |
|---|---|
| body | JSON object with `places` array, at most 500 entries, at most 512 KB |
| `id` | string matching `^[a-z0-9][a-z0-9-]{0,40}$`, unique in the list (case sensitive) |
| `name` | string, trimmed, 1 to 80 characters |
| `cat` | exactly one of `Furniture`, `Storage`, `Exits`, `Things` |
| `p` | exactly three finite numbers, `abs(x) < 5000` and `abs(z) < 5000` |
| `ai` | optional object: `cls` string 1 to 80 characters, `conf` finite number, `half` three finite numbers |
| `pinned` | optional boolean |

The API does not have per-scene bounds on this route, so the `p` check is the world bound above; the viewer clamps to
the scene's own bounds when it draws.

Unknown keys on a place or inside `ai` are preserved. The pipeline puts extras such as `ai.frames` there, and sending
back the list the app read does not lose them.

## Cache behaviour

The general `GET /v1/scenes/:id/:file` route caches whole files at Cloudflare's edge for an hour (`public,
max-age=3600`) because scene bytes are immutable and the bucket's free tier counts every download. Places are not
immutable, so they are explicitly exempt:

- `GET /v1/scenes/:id/places` never touches the edge cache (`caches.default`). It reads the bucket first, then Hugging
  Face when the pipeline published there, and answers `no-store`.
- `GET /v1/scenes/:id/places.json` through the old file route is also `no-store` now, so a client still on the file
  path sees a new pin immediately as well.
- A tag added now is visible on the next read, from any device.

## Where the file lives

The pipeline publishes scene files to Hugging Face when `PLY_HF_REPO` is set and to the bucket otherwise
(`pipeline/plypipe/worker.py`). A write through this route always goes to the bucket, so the read checks the bucket
first and Hugging Face second. An edited set is never hidden behind the copy the pipeline published. A scene id is a
job id and a re-run creates a new job, so no second places.json can appear under the same id.

## Examples

Read the pins for a scene:

```bash
curl -sS "https://api.example/v1/scenes/0ffc07fb-1111-2222-3333-444455556666/places" | jq
```

Add a pin, sending the current list plus the new place (the team key is a secret: keep it in the app, never in a
public page):

```bash
curl -sS -X PUT \
  -H "X-Ply-Key: $APP_KEY" \
  -H "Content-Type: application/json" \
  -d '{"places":[
        {"id":"desk","name":"Study desk","cat":"Furniture","p":[0.62,0.86,1.62],"pinned":true},
        {"id":"reading-nook","name":"Reading nook","cat":"Furniture","p":[1.9,0.4,-1.2]}
      ]}' \
  "https://api.example/v1/scenes/0ffc07fb-1111-2222-3333-444455556666/places"
```

The app should send the list it last read, with its own edit applied, so a rename or a move keeps the other pins.
