# AutoGuard AI — "Black Box" identity

The app is the car's own **black box**: it records a trip and understands it. The identity comes from a
flight recorder: an orange casing with reflective bands, graphite metal, and a screen with a live trace.
The concept board (three directions; B was chosen) is at
https://claude.ai/artifact/PwBt2RpwnR6zAWCwJuk5JT.

The earlier Figma file (navy identity, https://www.figma.com/design/RiA73c3OYNiGvJP1H7QUH2) is
**superseded** by this identity.

## Logo
- `app/assets/logo.svg` is the master: an orange casing with two clipped reflective bands, and a
  graphite screen with a white trace ending in the orange REC point.
- `app/assets/logo_small.svg` is for 16–32 px: no bands, bigger screen, bolder trace. The thin
  details blur at icon size.
- `python tools/make_brand_assets.py` renders `logo_16…512.png` and the multi-size `app.ico`.
- The logo is used for the EXE icon, the window/taskbar icon, the header, the home panel and the PDF
  report header.

## Colour (`app/theme.py`)
| token | value | use |
|---|---|---|
| INK (graphite) | #121417 | header, brand panel, **primary buttons** |
| ACCENT (recorder orange) | #FF6A13 | the one brand accent: active tab, REC tag, logo, progress, focus |
| ACCENT_TEXT | #B8470B | orange **text** and links (passes 4.5:1 on paper; #FF6A13 does not) |
| BG (paper) · CARD · BORDER | #F6F4EF · #FFFFFF · #E5E1D8 | content surfaces |
| TEXT · MUTED | #121417 · #5E6168 | text |
| OK · ATTENTION · WARNING | #1F7A4D · #A16207 · #B42318 | results; deep, calm tones |
| SYNTHETIC | #6D4AE0 | anything synthetic (demo / injected faults) |

## Type (bundled, `app/assets/fonts/`, SIL Open Font License)
- **IBM Plex Sans Arabic:** all UI text. It has both Arabic and Latin, so one face serves both
  languages.
- **Archivo ExtraBold:** the `AUTOGUARD` wordmark.
- **IBM Plex Mono:** the `• REC · AI` tag. It has no ● glyph, so the tag uses •.
- If the font files are missing, the stylesheet falls back to Segoe UI.

## Rules
- **Orange is precious:** one accent per area. Actions are graphite, not a second bright colour.
- **Orange text** always uses ACCENT_TEXT. Never use #FF6A13 for small text.
- **No emoji in the UI.** Use the line icons in `app/icons.py`: 24 px grid, 1.8 px round stroke.
- **No letter-spacing on Arabic text:** it breaks the joining between letters.
- Arabic is the default and the layout is right-to-left. Every screen must also work in English.
- In PDFs, use √ (not ✓) for checks: the Windows fonts have no ✓ glyph.
