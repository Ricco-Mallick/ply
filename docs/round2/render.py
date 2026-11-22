#!/usr/bin/env python3
"""Render the Round 2 documents to A4 PDF with the house page style.

    python3 docs/round2/render.py                 # ABSTRACT, SOP, MASTER-PDF, COST-ANALYSIS, VIDEO-SCRIPT, ATTRIBUTION
    python3 docs/round2/render.py ABSTRACT.md     # one file
    python3 docs/round2/render.py --count-only SOP.md

The Markdown file is the source. This turns it into HTML with the same type size and margins as docs/report.html
(9.35 pt Helvetica on A4, 14 mm / 15 mm / 13 mm), then prints it with the browser that made docs/report.pdf:

    google-chrome --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=...

`docs/report.pdf` and `docs/links.pdf` are byte-for-byte what that command writes today, so this is the same path.
Needs `markdown` (pip install markdown), `google-chrome` and `pdfinfo` (poppler-utils).

`<!-- page -->` on its own line starts a new page. Page counts are printed for every file: the abstract and the SOP
share the organisers' 5-page limit, and this is how the limit is checked.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROME = shutil.which("google-chrome") or shutil.which("google-chrome-stable") or shutil.which("chromium")

CSS = """
@page { size: A4; margin: 14mm 15mm 13mm; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body { font: 9.35pt/1.44 "Helvetica Neue", Helvetica, Arial, sans-serif; color: #14181d; background: #fff; }
h1, h2, h3 { margin: 0; font-weight: 700; letter-spacing: -0.01em; }
h1 { font-size: 15.5pt; line-height: 1.1; margin: 0 0 1.8mm; letter-spacing: -0.02em; }
h2 { font-size: 12.6pt; margin: 4.2mm 0 1.6mm; }
h3 { font-size: 10pt; margin: 3.2mm 0 1.2mm; }
h4 { font-size: 9.35pt; margin: 2.6mm 0 1mm; }
p { margin: 0 0 2.1mm; }
ul, ol { margin: 0 0 2.1mm; padding-left: 4.4mm; }
li { margin-bottom: .7mm; }
table { width: 100%; border-collapse: collapse; font-size: 8.15pt; margin: 1.4mm 0 2.4mm; }
th { text-align: left; font-weight: 600; color: #5b6572; border-bottom: 1px solid #c9d0d8; padding: .9mm 2mm .9mm 0; vertical-align: bottom; }
td { padding: .8mm 2mm .8mm 0; border-bottom: 1px solid #eceff2; vertical-align: top; }
td + td, th + th { padding-left: 1.6mm; }
td a, .url { word-break: break-all; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 8.1pt; letter-spacing: .01em; }
pre { margin: 1.4mm 0 2.4mm; padding: 1.6mm 2mm; background: #f7f9fb; border: 1px solid #e2e7ec; border-radius: 2px; page-break-inside: avoid; }
pre code { font-size: 7.35pt; line-height: 1.28; letter-spacing: 0; }
blockquote { margin: 1.6mm 0 2.2mm; border-left: 2px solid #ff6b00; padding: .6mm 0 .6mm 2.8mm; color: #333c46; }
blockquote p { margin: 0 0 1.4mm; }
blockquote p:last-child { margin: 0; }
hr { border: 0; height: 1px; background: #d8dde3; margin: 3.2mm 0; }
.brandline { display: flex; justify-content: space-between; font-size: 7.8pt; letter-spacing: .15em; text-transform: uppercase; color: #5b6572; border-bottom: 1px solid #d8dde3; padding-bottom: 1.8mm; margin-bottom: 4.2mm; }
.lead { font-size: 10pt; color: #333c46; }
.pagebreak { page-break-after: always; height: 0; }
h2, h3 { page-break-after: avoid; }
table, blockquote { page-break-inside: auto; }
"""

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title><style>{css}</style></head>
<body>
<div class="brandline"><span>Team Ply</span><span>{right}</span></div>
{title_html}
{body}
</body></html>
"""


def render_md(p: Path) -> str:
    try:
        import markdown  # type: ignore
    except ImportError:
        sys.exit("needs the `markdown` package: python3 -m pip install markdown")
    text = p.read_text(encoding="utf-8")
    # the first H1 becomes the document title, so it is not printed twice
    title = None
    m = re.match(r"#\s+(.+)\n", text)
    if m:
        title = m.group(1).strip()
        text = text[m.end():]
    text = re.sub(r"^<!--\s*page\s*-->$", '<div class="pagebreak"></div>', text, flags=re.M)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    body = body.replace("<p><div class=\"pagebreak\"></div></p>", '<div class="pagebreak"></div>')
    title_html = f"<h1>{title}</h1>" if title else ""
    return HTML.format(title=title or p.stem, css=CSS, right="Techfest 2026-27 · Namma Space · Round 2", title_html=title_html, body=body)


def build(p: Path, count_only: bool = False) -> int:
    html = render_md(p)
    html_path = p.with_suffix(".html")
    if not count_only:
        html_path.write_text(html, encoding="utf-8")
    if not CHROME:
        sys.exit("google-chrome not found")
    pdf = p.with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as td:
        args = [CHROME, "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                f"--user-data-dir={td}/profile", f"--print-to-pdf={pdf}", html_path.as_uri()]
        r = subprocess.run(args, capture_output=True, text=True)
        if not pdf.exists():
            sys.exit(f"chrome failed on {p.name}:\n{r.stderr[-1200:]}")
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", out).group(1)) if "Pages:" in out else -1
    return pages


def main(argv: list[str]) -> None:
    count_only = "--count-only" in argv
    names = [a for a in argv if not a.startswith("-")]
    if not names:
        names = ["ABSTRACT.md", "SOP.md", "MASTER-PDF.md", "COST-ANALYSIS.md", "VIDEO-SCRIPT.md", "ATTRIBUTION.md"]
    total = 0
    rows = []
    for n in names:
        p = HERE / n
        if not p.exists():
            print(f"skip {n} (not written yet)")
            continue
        pages = build(p, count_only)
        size = p.with_suffix(".pdf").stat().st_size // 1024 if p.with_suffix(".pdf").exists() else 0
        rows.append((n, pages, size))
        if n in ("ABSTRACT.md", "SOP.md"):
            total += pages
    for n, pages, size in rows:
        print(f"{n:20s} {pages:2d} page(s)  {size:5d} KB")
    if total:
        print(f"abstract + SOP:      {total:2d} page(s)  (organisers' limit: 5)")


if __name__ == "__main__":
    main(sys.argv[1:])
