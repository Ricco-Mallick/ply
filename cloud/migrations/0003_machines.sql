-- Paired PCs (src/machines.ts, pipeline/local/start.bat): each has its own token and takes only jobs assigned to it.
CREATE TABLE IF NOT EXISTS machines (
  name TEXT PRIMARY KEY,              -- also its worker name in workers / jobs.worker
  status TEXT NOT NULL DEFAULT 'pending',  -- pending (code shown on the PC) | approved (token not yet collected) | active | revoked
  pair_code TEXT,                     -- XXXX-XXXX shown on the PC while pending; NULL after
  poll_hash TEXT,                     -- sha256 of the PC's poll secret: only that PC can collect the token
  token_hash TEXT,                    -- sha256 of the machine secret once issued
  specs TEXT,                         -- JSON, latest plypipe.device report (GPU, RAM, what it can train)
  label TEXT,                         -- the team's own note ("college lab 3", "Toeesh's laptop")
  created_at INTEGER NOT NULL,
  approved_at INTEGER,
  last_seen INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS machines_code ON machines (pair_code) WHERE pair_code IS NOT NULL;
-- a job assigned to a machine by name: only that machine claims it, and Modal is never woken for it
ALTER TABLE jobs ADD COLUMN assigned_worker TEXT;
CREATE INDEX IF NOT EXISTS jobs_assigned ON jobs (assigned_worker, status);
