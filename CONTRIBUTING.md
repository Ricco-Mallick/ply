# Contributing

This repository is private and proprietary. Access is granted to specific people, and access
is not a licence: see LICENSE.

## Before your first pull request

1. Read LICENSE and CLA.md.
2. Sign the CLA by commenting on your first pull request with the line from CLA.md. Nothing is
   merged before that comment exists.

## Rules

- **Never commit secrets.** Keys live in `cloud/.secrets` and `cloud/.b2` (gitignored) and in
  platform secret stores. If one leaks, rotate it first and tell the owner.
- **No third-party code without a licence check.** MIT, BSD and Apache-2.0 are fine; add the
  component to NOTICE in the same pull request. Copyleft (GPL/AGPL) or non-commercial licences
  need the owner's written approval first.
- **Keep it tested.** `cd pipeline && .venv/bin/python -m pytest -q` and
  `cd app && npx tsc --noEmit && npx vite build` before you push.
- **Do not publish captures or twins.** Rooms captured for this project, their photos, point
  clouds, splats and the shells derived from them are private data.
- **Branches.** Work on a branch and open a pull request; `main` deploys the public site, so
  nothing lands there unreviewed.

## Ownership of contributions

By opening a pull request you agree that your contribution becomes the property of the owner,
as set out in CLA.md, and that you have the right to grant it.
