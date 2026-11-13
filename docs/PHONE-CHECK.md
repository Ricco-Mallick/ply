# The phone upload check (plan 7.4)

Plan 7.4 says a real clip from the day's camera goes through the phone upload path once. It is the last unmeasured
ship condition, and it needs a phone because the failure it is looking for only happens there: a phone records HEVC,
and a desktop browser decodes it while an Android one may not.

## What changed so the check can pass

`PROJECT-REVIEW.md` C1 was open: the capture app **dropped** a clip the browser could not decode, although the SOP
allows HEVC, and a clip with an unknown duration or a stalled seek **hung** with no cancel. Both are fixed on this
branch, so the check now tests the product rather than a known bug:

- an unreadable clip is kept with the metadata the file itself carries, and the app says the quality score was
  skipped instead of discarding footage;
- the analysis is bounded: 20 s overall and 2 s per seek, so a bad file cannot freeze the upload screen.

## The check, in three steps

1. **Record on the phone**, in the app: `https://ricco.codes/ply/capture.html?key=<team code>`, AR walk, about 60
   seconds, ending where you started. Record a second clip **on the native camera app** as well, so there is one
   HEVC file that did not come from the browser.
2. **Upload both** from the phone on the review screen. What to look for: two clips listed, no red error text, and if
   the native HEVC clip is the one the browser cannot read, a line saying it was kept without a quality score.
3. **Submit the job** and watch it reach `review`. A job that reaches `review` proves the upload, the queue, the owner
   column, the GPU run and the gate all carried a phone-recorded clip end to end.

## What passing looks like

- Both clips appear on the review screen.
- The job is created and has an owner of `team` (or the invite's label).
- It reaches `review` with a report; the twin opens in the walkthrough.

## What to do if it does not

The two failure modes this check exists to catch, and what each means:

- **A clip is missing after upload.** The fallback path failed to keep it: read the console on the phone with
  `chrome://inspect`, and the error text will name the file.
- **The upload screen never finishes.** The analysis budget did not apply: the fix bounds it at 20 s, so a longer
  hang means a different code path is seeking, and the console will show it.
