# AutoGuard AI — Brand & UI design

Figma file (brand, foundations, components, screens):
https://www.figma.com/design/RiA73c3OYNiGvJP1H7QUH2

The app (`app/theme.py`) and the Figma variables (`AutoGuard / Color`, `AutoGuard / Size`) use the
same values. Change a colour in both places.

## Logo
`app/assets/logo.svg` is the master. It shows a guard shield (protection) around an instrument gauge
(the car). A live signal trace (its telemetry) crosses it and ends in a cyan reading point (the AI
reading it).
`python tools/make_brand_assets.py` renders `logo_16…512.png` and the multi-size `app.ico`. The EXE
icon, the window/taskbar icon, the in-app header and the PDF report header all use these files.

## Colour
| token | value | use |
|---|---|---|
| brand/navy | #0B1626 | header, brand surfaces |
| brand/accent | #22D3EE | accent only: active tab, "AI" tag, logo point |
| action/primary | #1C4FD8 | the one main action per area |
| text/primary · text/muted | #0F1B2D · #5B6779 | text |
| bg/app · bg/card · border/default | #F3F5F9 · #FFFFFF · #E3E8EF | surfaces |
| status/ok · attention · warning | #0F7B55 · #B45309 · #C2283A | results; deep, calm tones |
| status/demo | #6D4AE0 | anything synthetic (demo / injected faults) |

## Rules
- **One primary button per area.** Other actions are secondary (outlined).
- **No emoji in the UI.** They render differently on each Windows version. Use the line icons in
  `app/icons.py`: 24 px grid, 1.8 px round stroke, drawn for this project.
- Status colour goes on soft backgrounds or a side bar, never as a full red frame.
- Arabic is the default and the layout is right-to-left. Every screen must also work in English
  (left-to-right).
- Type: Segoe UI in the app. Figma uses Inter and Noto Sans Arabic as stand-ins, because Segoe UI is
  not available there.

## Figma file notes
- It is built on the Starter plan, so it has 3 pages: *Brand & Foundations*, *Components* (Button,
  Status pill, Finding card, bound to the variables) and *Screens* (Home and Results, in Arabic).
- Known leftover: on the *Screens* page, in the Results header, the inactive tab labels "التقرير" and
  "الرحلة" render dark instead of `text/on-navy-muted`. The MCP call limit of the Starter plan was
  reached before this was fixed. Fix in Figma: select both labels → Fill → `text/on-navy-muted`.
