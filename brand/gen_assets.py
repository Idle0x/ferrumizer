"""Generate ferrumizer brand SVGs + PNG exports per brand/BUILD_SPEC.md v1.0.

Deliverables:
  mark.svg           primary symbol, full color, dark-bg variant
  mark-mono-dark.svg symbol only, single ink #23262A (field = outline only)
  favicon.svg        simplified small-scale symbol, viewBox 0 0 64 64
  lockup-dark.svg    symbol + wordmark + tagline on charcoal
  lockup-light.svg   symbol + wordmark + tagline on warm off-white

SVG is canonical; PNG exports (spec sizes + 512/1024) are for README/preview.

Geometry (200-unit canvas, center C=(100,100); screen coords, y down):
- outer instrument ring r=92: ONE arc path with round caps; the 20-45deg gap
  (clockwise from +x) IS the dial notch (315.05/350.05 deg long-way flags).
- secondary case-depth ring r=74, 85% opacity, hairline weight.
- material field r=58, off-center radial gradient (cx 34% cy 68% r 88%).
- profile + partial-derivative glyph: ONE continuous ~320-unit stroke
      rise (42,142) -> (84,110) -> cusp onto bowl
      bowl: exact circular arc  r=8.5 center (97,97), 310deg screen-CW sweep,
            OPEN 50deg gap in the upper-right (the partial counter);
            three cubic Bezier arcs, kappa = 4/3 tan(sweep/4),
            max radius deviation ~0.7%
      shoulder: bowl exit (101.25, 89.64) -> (110, 70)  [partial ascender]
      run-out:  (110,70) -> (157,72.5)  [flat hardened-case plateau]
- dashed 550 HV reference: y=115, 3-on/5-off, 50% cream, hairline.
- favicon: 0.32 scale; drops secondary ring, dashed ref, text.
- lockups: mark geometry inlined (not <image> refs) at translate(80,10);
  wordmark baseline 117, tagline 149 (optically mid-band, >=40u margins).
"""
import math
from pathlib import Path

INK_DARK = "#ECE7DB"
INK_LIGHT = "#23262A"
GOLD = "#D6B57C"
CHARCOAL = "#16181C"
OFFWHITE = "#F3F0EA"
CREAM = "#EFE6CF"

STOPS = ('<stop offset="0" stop-color="#C8A76E"/><stop offset=".26" '
         'stop-color="#A05F30"/><stop offset=".55" stop-color="#5D3320"/>'
         '<stop offset=".82" stop-color="#2B2723"/><stop offset="1" '
         'stop-color="#1D1F23"/>')


def grad(gid: str) -> str:
    return f'<radialGradient id="{gid}" cx="34%" cy="68%" r="88%">{STOPS}</radialGradient>'


def pol(deg, r=92.0):
    a = math.radians(deg - 90)
    return (100 + r * math.cos(a), 100 + r * math.sin(a))


N0, N1 = pol(45), pol(20)
# sweep=1 (the long way: 45deg -> 360 -> 20deg), large-arc=1
OUTER = f"M {N0[0]:.2f} {N0[1]:.2f} A 92 92 0 1 1 {N1[0]:.2f} {N1[1]:.2f}"

PROFILE = (
    "M 42 142 "
    "C 58 138 73 126 84 110 "
    "C 89.5 104.5 99 100.5 105.37 95.52 "
    "C 106.0 100.0 103.5 103.0 101.25 104.36 "
    "C 95.4 107.2 90.5 105.5 89.64 97.01 "
    "C 88.0 90.2 95.5 85.5 101.25 89.64 "
    "C 106.5 84.5 110.5 78.0 110.0 70.0 "
    "C 122 69 141 70.8 157 72.5"
)
REF = "M 43 115 H 157"


def mark_body(ink: str, gid: str, mono: bool = False) -> list[str]:
    if mono:
        field = f'<circle cx="100" cy="100" r="58" fill="none" stroke="{ink}" stroke-width="2.2"/>'
        ref_col, ref_op = ink, ".45"
        prof = ink
    else:
        field = f'<circle cx="100" cy="100" r="58" fill="url(#{gid})" stroke="{ink}" stroke-width="1.3"/>'
        ref_col, ref_op = CREAM, ".5"
        prof = GOLD
    return [
        f'<path d="{OUTER}" fill="none" stroke="{ink}" stroke-width="4" stroke-linecap="round"/>',
        f'<circle cx="100" cy="100" r="74" fill="none" stroke="{ink}" stroke-opacity=".85" stroke-width="1.3"/>',
        field,
        f'<path d="{PROFILE}" fill="none" stroke="{prof}" stroke-width="3.8" '
        f'stroke-linecap="round" stroke-linejoin="round"/>',
        f'<path d="{REF}" fill="none" stroke="{ref_col}" stroke-opacity="{ref_op}" '
        f'stroke-width="1.2" stroke-dasharray="3 5" stroke-linecap="round"/>',
    ]


def mark_svg(ink: str, mono: bool = False, gid: str = "field") -> str:
    defs = "" if mono else f"<defs>{grad(gid)}</defs>"
    lines = [ln for ln in [defs, *mark_body(ink, gid, mono)] if ln]
    body = "\n".join("  " + ln for ln in lines)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">\n{body}\n</svg>\n'


def favicon() -> str:
    g = grad("f")
    ring_w = 3.4 / 0.32
    prof_w = 3.2 / 0.32
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">\n'
        f"  <defs>{g}</defs>\n"
        '  <g transform="translate(32 32) scale(0.32) translate(-100 -100)">\n'
        f'    <path d="{OUTER}" fill="none" stroke="{INK_DARK}" stroke-width="{ring_w:.1f}" stroke-linecap="round"/>\n'
        f'    <circle cx="100" cy="100" r="58" fill="url(#f)" stroke="{INK_DARK}" stroke-width="3.75"/>\n'
        f'    <path d="{PROFILE}" fill="none" stroke="{GOLD}" stroke-width="{prof_w:.1f}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>\n'
        "  </g>\n"
        "</svg>\n"
    )


FONT = "'Inter','SF Pro Text','Segoe UI','Helvetica Neue',Arial,sans-serif"


def lockup(ink: str, bg: str, gid: str) -> str:
    body = "\n    ".join(mark_body(ink, gid, mono=False))
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 220">\n'
        f"  <defs>{grad(gid)}</defs>\n"
        f'  <rect width="640" height="220" fill="{bg}"/>\n'
        '  <g transform="translate(80 10)">\n'
        f"{body}\n"
        "  </g>\n"
        f'  <text x="322" y="117" fill="{ink}" font-family="{FONT}" font-size="54" font-weight="600" '
        f'letter-spacing=".5">ferrumizer</text>\n'
        f'  <text x="324" y="149" fill="{ink}" fill-opacity=".6" font-family="{FONT}" font-size="16" '
        f'font-weight="400" letter-spacing="1.3">Gradients through the furnace</text>\n'
        "</svg>\n"
    )


def main():
    out = Path(__file__).resolve().parent
    (out / "mark.svg").write_text(mark_svg(INK_DARK, gid="field"))
    (out / "mark-mono-dark.svg").write_text(mark_svg(INK_LIGHT, mono=True))
    (out / "favicon.svg").write_text(favicon())
    (out / "lockup-dark.svg").write_text(lockup(INK_DARK, CHARCOAL, "field-d"))
    (out / "lockup-light.svg").write_text(lockup(INK_LIGHT, OFFWHITE, "field-l"))

    import cairosvg

    for name, w in (("mark", 600), ("mark-mono-dark", 400), ("favicon", 256),
                    ("lockup-dark", 1280), ("lockup-light", 1280)):
        cairosvg.svg2png(url=str(out / f"{name}.svg"),
                         write_to=str(out / f"{name}.png"), output_width=w)
    for name in ("mark", "mark-mono-dark", "favicon", "lockup-dark", "lockup-light"):
        for w in (512, 1024):
            cairosvg.svg2png(url=str(out / f"{name}.svg"),
                             write_to=str(out / f"{name}-{w}.png"), output_width=w)
    print("wrote 5 SVGs + PNG exports")


if __name__ == "__main__":
    main()
