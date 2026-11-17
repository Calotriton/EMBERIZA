# 0005. The name EMBERIZA, and a dark "embers" look
Status: Accepted
Date: 2026-11-16

## Context

The owner asked for a name that is an acronym for a bird or amphibian, with an E for Embeddings and a B for bacpipe, to be confirmed by them. They asked for a modern, technical look, because embeddings are a new tool in bioacoustic monitoring. The page must name bacpipe as the work it runs, and must not suggest that it is bacpipe or endorsed by its authors.

## Options considered

For the name, offered to the owner:
1. **EMBERIZA**, the genus of the buntings: **E**mbedding **M**aps for **B**acpipe: **E**xplore **R**ecordings **I**n a **Z**ero-install **A**pp.
2. Other bird and amphibian names, offered at the same time.

For the look:
1. **Dark first, with a light theme.** Charcoal ground, a faint grid, ember orange for actions and the selected state, a cool cyan as the second colour. Numbers in a monospaced face. The welcome card shows a field of points drifting like a map of embeddings.
2. **Light first,** like Apus.
   - Loses: less of the technical feel asked for.

## Decision

- **Name:** EMBERIZA, chosen by the owner on 2026-11-16. "Ember" also gives the colours.
- **Look:** option 1.
  - Colours are tokens in `app/style.css` (`--bg #0b0e13`, `--accent #ff8a4c`, `--cool #5cc8e8`), redefined for `[data-theme="light"]`. Dark is the default. The button in the header switches, and the browser remembers the choice.
  - System fonts only (`system-ui` and `ui-monospace`), so no font is downloaded.
  - The logo (`app/icons/logo.svg`) was a bunting drawn as a constellation of ember points. ADR 0007 replaced it with a pixel-art reed bunting.
  - The maps use Plotly's WebGL scatter. The continuous scales and the spectrogram colours are the ones bacpipe's settings name.
  - Motion is switched off for visitors who ask for reduced motion.
- **bacpipe is credited everywhere it matters:** in the header link, on the welcome card with the paper, in About with the licences, and in the README. Every page that could be mistaken for bacpipe's says "EMBERIZA is an independent interface and is not part of bacpipe".

## Consequences

- **Easier:** a recognisable page, and a name that tells what it does.
- **Harder:** two themes to check in every change of the interface.
- **Reversal:** the colours are tokens, so a new palette is a change of one block of CSS.
