# Design System — Heidi Health Reference

Extracted from [heidihealth.com/en-us](https://www.heidihealth.com/en-us).

---

## Color Palette

### Brand Colors

| Name    | Light / Default         | Dark Variant  |
|---------|-------------------------|---------------|
| Bark (Primary) | `#28030f`          | —             |
| Sand (Accent)  | `#f9f4f1`          | `#f6ece4`     |
| Forest (Secondary) | `#194b22`      | `#407648`     |
| Sky     | `#5b8df6`               | `#072c7e`     |
| Sunlight | `#fbf582`              | `#4d4900`     |

### Semantic / UI Colors

| Role    | Hex       |
|---------|-----------|
| Warning | `#ea580c` |
| Danger  | `#dc2626` |
| Success | `#16a34a` |
| Emerald | `oklch(59.6% .145 163.225)` |

### Neutral Grays (OKLch)

| Token    | Value                        |
|----------|------------------------------|
| Gray-900 | `oklch(21% .034 264.665)`    |
| Gray-700 | `oklch(37.3% .034 259.733)`  |
| Gray-600 | `oklch(44.6% .03 256.802)`   |
| Gray-300 | `oklch(87.2% .01 258.338)`   |

> Color space: OKLch (perceptually uniform). Primary dark is a deep warm brown-black (`#28030f`); backgrounds use warm sand tones.

---

## Typography

### Font Families

| Role        | Stack                                                   |
|-------------|----------------------------------------------------------|
| Display     | `Exposure` (serif)                                       |
| Body/UI     | `ui-sans-serif, system-ui, sans-serif`                   |
| Monospace   | `ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas`  |
| Arabic      | `Noto Sans Arabic`                                       |

### Heading Scale (Exposure — weight 400)

| Tag | Size      |
|-----|-----------|
| H1  | `4.5rem`  |
| H2  | `3.5rem`  |
| H3  | `3rem`    |
| H4  | `2.5rem`  |
| H5  | `2rem`    |
| H6  | `1.5rem`  |

### Body Scale

| Token | Size       |
|-------|------------|
| Body  | `1rem`     |
| Small | `0.875rem` |
| XS    | `0.75rem`  |

### Font Weights

| Name      | Value |
|-----------|-------|
| Light     | 300   |
| Normal    | 400   |
| Medium    | 500   |
| Semibold  | 600   |
| Bold      | 700   |

### Spacing & Rhythm

| Property        | Value      |
|-----------------|------------|
| Heading tracking | `-0.05em` |
| Body tracking    | `-0.03em` |
| Body line-height | `140%`    |
| Tight line-height | `1.25`   |
| Relaxed line-height | `1.625` |

---

## Spacing System

Base unit: `0.25rem` (4px).

| Scale Token     | Value   |
|-----------------|---------|
| Section large   | `7rem`  |
| Section medium  | `5rem`  |
| Section small   | `3rem`  |
| Global spacing  | `4rem`  |

All margin/padding values are multiples of the `0.25rem` base.

---

## Border & Radius

| Token | Value     |
|-------|-----------|
| XS    | `4px`     |
| SM    | `6px`     |
| MD    | `8px`     |
| LG    | `12px`    |
| XL    | `16px`    |
| 2XL   | `24px`    |
| 3XL   | `36px`    |
| Full  | pill / fully rounded |

Border widths: `1px`, `2px`, `4px`. Default style: `solid`.

---

## Shadows

| Token        | Value                                                              |
|--------------|--------------------------------------------------------------------|
| XS           | `0 1px 3px 1px rgb(120 90 60 / .06), 0 1px 1px 0 rgb(120 90 60 / .03)` |
| Drop-LG      | `0 4px 4px #00000026`                                              |
| Drop-XL      | `0 9px 7px #0000001a`                                              |

---

## Animations & Transitions

**Default easing:** `cubic-bezier(.4, 0, .2, 1)` at `0.15s`

| Name       | Definition                              |
|------------|-----------------------------------------|
| Fade-in    | `0.5s ease-in`                          |
| Fade-in-up | `0.3s ease-out`                         |
| Breathing  | `10s ease-in-out infinite`              |
| Ping       | `1s cubic-bezier(0,0,.2,1) infinite`   |
| Pulse      | `2s cubic-bezier(.4,0,.6,1) infinite`  |
| Spin       | `1s linear infinite`                    |

---

## Layout System

### Container Widths

| Token | Value    |
|-------|----------|
| 3XS   | `15rem`  |
| XS    | `25rem`  |
| SM    | `30rem`  |
| MD    | `35rem`  |
| LG    | `48rem`  |
| XL    | `64rem`  |
| 2XL   | `42rem`  |
| Max   | `80rem`  |

### Grid

- 12-column grid system
- Flexbox utilities: row, column, wrap, gap (multiples of base unit)
- Common aspect ratios: `16/9`, `4/3`, `2/1`

---

## Component Conventions

- **Buttons:** `cursor-pointer`; disabled state uses `cursor-not-allowed`
- **Inputs:** `appearance: none`; touch targets use `touch-action: manipulation`
- Interactive elements follow Tailwind-style utility patterns

---

## Design Character

- **Tone:** Warm, earthy, minimal — not clinical
- **Contrast:** Deep bark brown (`#28030f`) against sand/cream backgrounds
- **Hierarchy:** Serif display (Exposure) for headings; clean system sans-serif for body
- **Feel:** Human and trustworthy; rounded corners, soft shadows, subtle warm tints
