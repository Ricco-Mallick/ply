-- Per-invite ownership, idempotent job creation, and the publish gate (plan phase 0.3 and 0.4).
ALTER TABLE jobs ADD COLUMN owner TEXT;      -- invite label, or 'team' for the app key
ALTER TABLE jobs ADD COLUMN idem TEXT;        -- client-supplied idempotency key, nullable
CREATE UNIQUE INDEX IF NOT EXISTS jobs_idem ON jobs (idem) WHERE idem IS NOT NULL;
CREATE INDEX IF NOT EXISTS jobs_owner ON jobs (owner, created_at);
