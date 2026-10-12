-- Ply job queue (Cloudflare D1). One row per processing job; the GPU worker claims the oldest queued row.
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  capture_id TEXT NOT NULL,
  settings TEXT NOT NULL,          -- JSON: preset + advanced
  args TEXT NOT NULL,              -- JSON: plypipe flags
  files TEXT NOT NULL,             -- JSON: R2 keys of the clips + capture.json
  status TEXT NOT NULL DEFAULT 'queued',   -- queued | running | done | failed
  stage TEXT,                      -- frames | sfm | align | occupancy | train | export
  progress REAL NOT NULL DEFAULT 0,
  message TEXT,
  worker TEXT,
  result TEXT,                     -- JSON: { base, files, report } when done
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  started_at INTEGER,
  finished_at INTEGER
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (status, created_at);

-- GPU workers check in on every poll; the status page shows whether one is online
CREATE TABLE IF NOT EXISTS workers (
  name TEXT PRIMARY KEY,
  last_seen INTEGER NOT NULL,
  busy_job TEXT
);
