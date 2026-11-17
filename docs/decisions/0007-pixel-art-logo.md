# 0007. The logo is a pixel-art reed bunting
Status: Accepted (replaces the logo of ADR 0005)
Date: 2026-11-17

## Context

ADR 0005 gave EMBERIZA a logo of ember points in the shape of a bunting. The owner asked for a new one: pixel art in a 16-bit style, showing *Emberiza schoeniclus*, the reed bunting, with the choice of face, whole body or another view left open.

The logo appears in two places: the header, at about 40 px, and the browser tab, at 16 px. Pixel art stays crisp only when each art pixel covers a whole number of screen pixels.

## Options considered

1. **Head and shoulders of a male in breeding plumage.** The reed bunting's field marks are all on the head and neck: black head and bib, a white moustache stripe that joins a white collar, and a streaked chestnut back.
   - Gains: the bird is recognisable at 16 px, where the head is about 9 pixels wide.
   - Loses: the long tail and the reed are not shown.
2. **The whole bird perched on a reed.**
   - Gains: the habitat that gives the species its name.
   - Loses: at 40 px the head is about 6 pixels, the moustache and collar shrink to single pixels, and at 16 px the bird is a brown spot.
3. **The head only.**
   - Loses: the chestnut back, the second colour that tells it from other black-headed birds.

## Decision

Option 1, facing right, towards the name.
- **Two drawings,** each at a size that scales by whole numbers:
  - `app/icons/logo.svg`: 20 x 20 pixels, shown at 40 x 40 px (2 x) in the header.
  - `app/icons/favicon.svg`: 16 x 16 pixels, a simplified version for the tab (1 x).
- **Style:** a limited palette with two or three shades per material, as on 16-bit consoles. The wing is drawn as diagonal feathers with dark centres and chestnut fringes, and a chestnut shoulder.
- **Tile:** ember orange (`#ff8a4c`, the page's accent) with cut corners. The black head needs a light ground, and the tile stands out on both themes and on dark and light tab bars.
- **Source:** the pixels are text maps in `tools/make_logo.py`, one character per pixel. The script writes both SVG files, one `<rect>` per run of equal colour with `shape-rendering="crispEdges"`. It can also write enlarged PNG previews.
- The drawing is original. It follows the bird's plumage, not any existing picture or logo.

## Evidence

- Screenshots of the header at 1 x and 1.5 x pixel density, in the dark and light themes: the head, moustache, collar and wing are distinct at 40 px.
- The 16 px icon on dark and light tab bars, at 1 x and 2 x: the black head, the white collar and the brown wing stay distinct.

## Consequences

- **Easier:** a logo that names the bird, in a style that suits a technical tool. Changing a pixel is changing a character.
- **Harder:** two drawings to keep alike.
- **Reversal:** the old SVG files are in the repository's history.
