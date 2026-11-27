#!/usr/bin/env bash
# plylocal: run Ply on this machine (WSL2 on a Windows PC, or plain Ubuntu). Installed by install_local.sh.
#
#   plylocal doctor                         what this machine can do (GPU, RAM, disk, the plan's ceilings)
#   plylocal run <clips|folder> [--name N] [--preset quick|standard|high] [plypipe run flags...]
#                                           whole pipeline here; resumable: the same --name picks up where it stopped
#   plylocal status [name]                  last events and the quality numbers of a job
#   plylocal results <name> [dest]          copy the twin files out (default on WSL: %USERPROFILE%\Ply\<name>)
#   plylocal jobs                           list jobs in $PLY_WORK
#   plylocal pair [--again]                 pair this PC with the Ply API (shows a code for the team to approve)
#   plylocal worker                         run the jobs the team assigns to this PC (pairs first if needed)
#   plylocal update                         git pull (fast-forward only) in the repo copy this runs from
#   plylocal forget                         borrowed machine, before handing it back: delete jobs, caches and the install
set -euo pipefail

[ -f "$HOME/.ply/env.sh" ] && . "$HOME/.ply/env.sh"
: "${PLY_PY:?not installed here: bash pipeline/local/install_local.sh}"
WORK="${PLY_WORK:-$HOME/ply-work}"
PIPE="$PLY_REPO/pipeline"
RESULT_FILES=(scene.spz scene_lite.spz scene.json occupancy.png occupancy.json topview.webp places.json shell.json report.json plan.json log.jsonl)
say() { printf '\033[1m[plylocal]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[plylocal] %s\033[0m\n' "$*" >&2; exit 1; }
is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }
win_home() {  # Windows' own PATH is not appended inside the Ply distro, so cmd.exe is called by its path
  local cmd=/mnt/c/Windows/System32/cmd.exe
  is_wsl && [ -x "$cmd" ] && wslpath -u "$(cd /mnt/c && "$cmd" /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')" 2>/dev/null
}

cmd_doctor() {
  (cd "$PIPE" && "$PLY_PY" -m plypipe.device --dir "$WORK" 2>/dev/null || "$PLY_PY" -m plypipe.device)
  local ok=1
  command -v nvcc >/dev/null && say "nvcc $(nvcc --version | grep -o 'release [0-9.]*')" || { say "nvcc missing"; ok=0; }
  command -v splat-transform >/dev/null && say "splat-transform $(splat-transform --version 2>/dev/null | head -1)" || { say "splat-transform missing (export will use npx)"; }
  "$PLY_TRAIN_PYTHON" -c "import torch, gsplat, fused_ssim; print('[plylocal] train env: torch', torch.__version__, 'gsplat', gsplat.__version__, 'cuda', torch.cuda.is_available())" \
    || { say "training environment broken: re-run install_local.sh"; ok=0; }
  command -v ffmpeg >/dev/null || { say "ffmpeg missing"; ok=0; }
  [ "$ok" = 1 ] && say "ready" || die "not ready"
}

cmd_run() {
  [ $# -ge 1 ] || die "usage: plylocal run <clip|folder> [--name N] [--preset P] [plypipe flags]"
  local src="$1"; shift
  [ -e "$src" ] || die "no such file or folder: $src"
  local name="" preset="standard" extra=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --name) name="$2"; shift 2;;
      --preset) preset="$2"; shift 2;;
      --max-frames|--max-frames=*) die "--max-frames lowers the frame rate and costs the twin its views; the plan picks frames from the clips (docs/PROCESSING.md)";;
      *) extra+=("$1"); shift;;
    esac
  done
  [ -n "$name" ] || name="$(basename "${src%/}" | sed 's/\.[^.]*$//; s/[^A-Za-z0-9._-]/-/g')"
  local input="$WORK/inputs/$name" out="$WORK/jobs/$name"
  mkdir -p "$WORK/inputs" "$WORK/jobs"
  # Windows drives (/mnt/c) are slow to read through WSL and may move or disappear before a resume: copy in once
  if [ ! -e "$input" ]; then
    say "copying input into $input"
    if [ -d "$src" ]; then mkdir -p "$input" && cp -r "${src%/}/." "$input/"; else mkdir -p "$input" && cp "$src" "$input/"; fi
  else
    say "input already here ($input): resuming"
  fi
  local need; need=$(du -sm "$input" | cut -f1)
  local free; free=$(df -Pm "$WORK" | awk 'NR==2 {print $4}')
  [ "$free" -gt $(( need * 6 + 10240 )) ] || die "only ${free} MB free for ${need} MB of clips: a job needs ~6x the clips plus 10 GB"
  say "job $name  preset $preset  ->  $out"
  cd "$PIPE"
  set +e
  "$PLY_PY" -m plypipe run "$input" --out "$out" --preset "$preset" "${extra[@]}"
  local code=$?
  set -e
  cmd_status "$name" || true
  [ "$code" = 0 ] || die "stopped (exit $code). Fix the cause above and run the same command again: finished stages are kept"
  say "done. Copy it out: plylocal results $name"
}

cmd_status() {
  local name="${1:-}"
  if [ -z "$name" ]; then name=$(ls -t "$WORK/jobs" 2>/dev/null | head -1); [ -n "$name" ] || die "no jobs yet"; fi
  local out="$WORK/jobs/$name"
  [ -d "$out" ] || die "no job $name in $WORK/jobs"
  say "job $name"
  [ -f "$out/log.jsonl" ] && "$PLY_PY" - "$out/log.jsonl" <<'PY'
import json, sys
lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()[-8:]
for line in lines:
    try:
        e = json.loads(line)
    except ValueError:
        continue
    print("  %-10s %-9s %s" % (e.get("stage", ""), e.get("event", ""), str(e.get("msg", ""))[:150]))
PY
  [ -f "$out/report.json" ] && "$PLY_PY" - "$out" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
r = json.loads((out / "report.json").read_text())
p = json.loads((out / "plan.json").read_text()) if (out / "plan.json").exists() else {}
fr, sfm, tr, ex = (r.get(k) or {} for k in ("frames", "sfm", "train", "export"))
print("  frames kept", fr.get("kept"), "| registered", sfm.get("registeredRatio"))
if tr:
    ev = tr.get("eval") or {}
    print("  splats", (tr.get("plan") or {}).get("capMax"), "| peak VRAM", tr.get("peakVramGB"), "GB | held-out PSNR",
          ev.get("ccPsnr") or ev.get("psnr"), "| antialiased", tr.get("antialiased"), "| OOM retry" if tr.get("oomRetry") else "")
if ex:
    print("  export kept", ex.get("keptFraction"), "| SPZ MB", ex.get("spzMB"))
if p.get("device"):
    print("  device:", p["device"].get("why"))
PY
}

cmd_results() {
  local name="${1:?usage: plylocal results <name> [dest]}" dest="${2:-}"
  local out="$WORK/jobs/$name"
  [ -d "$out" ] || die "no job $name"
  if [ -z "$dest" ]; then
    local wh; wh=$(win_home || true)
    dest="${wh:-$HOME}/Ply/$name"
  fi
  mkdir -p "$dest"
  local n=0
  for f in "${RESULT_FILES[@]}"; do [ -f "$out/$f" ] && cp "$out/$f" "$dest/" && n=$((n + 1)); done
  [ -f "$out/scene.ply" ] && [ "${PLY_COPY_PLY:-0}" = 1 ] && cp "$out/scene.ply" "$dest/"
  say "copied $n files to $dest$(is_wsl && printf ' (Windows: %s)' "$(wslpath -w "$dest" 2>/dev/null)")"
}

cmd_jobs() {
  [ -d "$WORK/jobs" ] || { say "no jobs yet"; return; }
  for d in "$WORK/jobs"/*/; do
    local n; n=$(basename "$d")
    local last; last=$(tail -n 1 "$d/log.jsonl" 2>/dev/null | "$PLY_PY" -c 'import json,sys; e=json.loads(sys.stdin.read() or "{}"); print(e.get("stage",""), e.get("event",""))' 2>/dev/null || true)
    printf '  %-32s %-24s %s\n' "$n" "$last" "$(du -sh "$d" | cut -f1)"
  done
}

cmd_pair() {
  cd "$PIPE"
  "$PLY_PY" -m plypipe.enroll "$@"
}

cmd_worker() {
  # a paired PC uses its own token from ~/.ply/worker.json (plypipe.enroll); a team box may still pass PLY_WORKER_TOKEN
  if [ -z "${PLY_WORKER_TOKEN:-}" ]; then
    (cd "$PIPE" && "$PLY_PY" -m plypipe.enroll --check >/dev/null) || cmd_pair
  fi
  [ -n "${PLY_WORKER_TOKEN:-}" ] && export PLY_API="${PLY_API:-https://ply-api.ricco-ply.workers.dev}"
  export PLY_WORK_DIR="${PLY_WORK_DIR:-$WORK/worker}"
  export PLY_IDLE_EXIT_MIN="${PLY_IDLE_EXIT_MIN:-0}"
  # a local GPU costs nothing per minute: the stage limits only stop a stuck run, so they are hours, not the cloud's minutes
  export PLY_STAGE_LIMITS="${PLY_STAGE_LIMITS:-frames=60,sfm=600,label=120,train=1440,export=180}"
  export PLY_JOB_MAX_MIN="${PLY_JOB_MAX_MIN:-2880}"
  say "worker starting; it runs only the jobs the Ply team assigns to this PC. Ctrl+C (or closing the window) stops it"
  cd "$PIPE"
  while true; do
    set +e
    "$PLY_PY" -m plypipe.worker
    local code=$?
    set -e
    [ "$code" = 3 ] && die "pairing revoked: run start.bat (or plylocal pair --again) to pair this PC again"
    [ "$code" = 0 ] && exit 0
    say "worker stopped (exit $code); restarting in 60 s"
    sleep 60
  done
}

cmd_update() {
  git -C "$PLY_REPO" pull --ff-only
  say "now at $(git -C "$PLY_REPO" log --oneline -1)"
}

cmd_forget() {
  say "this deletes: $WORK, ~/.ply, ~/.cache/huggingface, ~/.cache/uv, ~/.cache/pip, ~/.cache/torch_extensions, and the env line in ~/.bashrc"
  say "the repo copy at $PLY_REPO stays (delete it too with: rm -rf $PLY_REPO); on Windows also run: ply.ps1 forget"
  read -rp "[plylocal] type FORGET to go on: " ok
  [ "$ok" = FORGET ] || die "nothing deleted"
  local me; me=$(cd "$PIPE" && "$PLY_PY" -m plypipe.enroll --check 2>/dev/null | sed -n 's/^paired as //p' || true)
  rm -rf "$WORK" "$HOME/.ply" "$HOME/.cache/huggingface" "$HOME/.cache/uv" "$HOME/.cache/pip" "$HOME/.cache/torch_extensions"
  sed -i '/\.ply\/env\.sh/d' "$HOME/.bashrc"
  say "done${me:+. Ask the Ply team to revoke this PC too: python scripts/plyctl.py machines revoke $me}"
}

case "${1:-help}" in
  doctor) shift; cmd_doctor "$@";;
  run) shift; cmd_run "$@";;
  status) shift; cmd_status "$@";;
  results) shift; cmd_results "$@";;
  jobs) shift; cmd_jobs;;
  worker) shift; cmd_worker;;
  pair) shift; cmd_pair "$@";;
  update) shift; cmd_update;;
  forget) shift; cmd_forget;;
  *) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//';;
esac
