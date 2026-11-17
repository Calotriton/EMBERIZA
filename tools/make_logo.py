"""Draws EMBERIZA's logo: a male reed bunting (Emberiza schoeniclus) in breeding plumage, as 16-bit pixel art.

    python tools/make_logo.py [preview folder]

Writes app/icons/logo.svg (20 x 20 pixels, shown at 40 px in the header) and app/icons/favicon.svg (16 x 16, for the
browser tab). Each pixel is a character in the maps below, and each character a colour in PALETTE. With a folder,
it also writes PNG previews, enlarged and at their real sizes, on the page's dark and light backgrounds.
"""
import struct
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PALETTE = {
    ".": None,          # outside the tile (the cut corners)
    "~": "#ff8a4c",     # tile: ember orange (the page's accent)
    "K": "#141418",     # black head and bib
    "k": "#34363f",     # sheen on the crown
    "E": "#f2efe8",     # catch-light of the eye
    "W": "#f6f2ea",     # white moustache, collar and breast
    "w": "#d9d3c8",     # white in shade
    "g": "#a9a196",     # breast, deeper shade
    "Y": "#4a4c55",     # upper mandible
    "y": "#8c8f99",     # lower mandible
    "B": "#8a5734",     # back, brown
    "R": "#b8693a",     # chestnut (shoulder, feather edges)
    "r": "#d9a066",     # buff feather edges
    "D": "#2b1b12",     # dark streaks and outline of the back
    "s": "#8f7a66",     # faint flank streak
}

LOGO = """
..~~~~~~~~~~~~~~~~..
.~~~~~~~~~~~~~~~~~~.
~~~~~~~~KKKKK~~~~~~~
~~~~~~KKKkkKKKK~~~~~
~~~~~KKkKKKKKKKK~~~~
~~~~~KKKKKKKKEKK~~~~
~~~~~KKKKKKKKKKKYYY~
~~~~~KKKKKKKKKKKyy~~
~~~~~KKKKKKWWWKK~~~~
~~~~WWWKWWWWKKKK~~~~
~~~DBRWWWWWWKKKw~~~~
~~DBDRBwWWWWKKWw~~~~
~DBDRBDRwWWWKWWWw~~~
DBDRBRRRRwWWWWWWWw~~
BDBRRRRDDDwWWWWWWw~~
DDRBRDDRBDswWWWWWw~~
DRBRDDRBRDswWWWWWg~~
RBRDDRBRDDwswWWWwg~~
.RDDRBRDDDswwWWWg~~.
..DRBRDDRDgswwWwg~..
"""

FAVICON = """
.~~~~~~~~~~~~~~.
~~~~~~KKKK~~~~~~
~~~~~KKkKKKK~~~~
~~~~KKKKKKEKK~~~
~~~~KKKKKKKKKYY~
~~~~KKKKKKKKKy~~
~~~~KKKKKWWKK~~~
~~~WWWKWWWKKK~~~
~~DBWWWWWWKKw~~~
~DBDRwWWWWKWWw~~
DBDRRRwWWWWWWw~~
DRBRDDRDwWWWWWw~
RBRDDRBDswWWWWw~
BRDDRBRDswWWWWg~
RDDRBRDDwswWWwg~
.DRBRDDDswwwg~~.
"""


def grid(art):
    rows = [r for r in art.strip("\n").split("\n")]
    width = len(rows[0])
    assert all(len(r) == width for r in rows), [len(r) for r in rows]
    assert all(c in PALETTE for r in rows for c in r), {c for r in rows for c in r} - set(PALETTE)
    return rows


def svg(rows, title):
    """One <rect> per run of equal colour in a row, with crisp edges."""
    h, w = len(rows), len(rows[0])
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w * 2}" height="{h * 2}" '
           f'shape-rendering="crispEdges" role="img" aria-label="{title}">']
    for y, row in enumerate(rows):
        x = 0
        while x < w:
            end = x
            while end < w and row[end] == row[x]:
                end += 1
            if PALETTE[row[x]]:
                out.append(f'<rect x="{x}" y="{y}" width="{end - x}" height="1" fill="{PALETTE[row[x]]}"/>')
            x = end
    out.append("</svg>")
    return "\n".join(out) + "\n"


def png(rows, scale, background):
    """A PNG of the art, enlarged by `scale` with square pixels, on a plain background (for the previews)."""
    def rgb(hexa):
        return bytes(int(hexa[i:i + 2], 16) for i in (1, 3, 5))
    h, w = len(rows), len(rows[0])
    lines = []
    for row in rows:
        line = b"".join((rgb(PALETTE[c]) if PALETTE[c] else rgb(background)) * scale for c in row)
        lines += [line] * scale
    raw = b"".join(b"\x00" + line for line in lines)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w * scale, h * scale, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    logo, favicon = grid(LOGO), grid(FAVICON)
    (ROOT / "app/icons/logo.svg").write_text(svg(logo, "EMBERIZA"), encoding="utf-8")
    (ROOT / "app/icons/favicon.svg").write_text(svg(favicon, "EMBERIZA"), encoding="utf-8")
    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in (("logo", logo), ("favicon", favicon)):
            for theme, bg in (("dark", "#0b0e13"), ("light", "#f6f7f9")):
                for scale in (1, 2, 16):
                    (out / f"{name}-{theme}-{scale}x.png").write_bytes(png(rows, scale, bg))
    print("logo.svg", len(logo[0]), "x", len(logo), "| favicon.svg", len(favicon[0]), "x", len(favicon))
