"""
Builds the IC-SHM 2026 Project 2 presentation deck (paper/presentation/slides.pptx) from
paper/DRAFT.md's actual content and the real figures already generated for the paper. Every
number on every slide is copied from DRAFT.md verbatim (see the `# DRAFT.md <line>` comments
scattered through this file) rather than retyped from memory, so re-running this script after a
paper edit is the way to keep the deck in sync, not a manual slide-by-slide patch.

Layout/visual style loosely follows paper/slides-ref.pdf (a reference deck for a different
project in this same competition): a solid-color title bar per slide, a light background,
numbered-circle bullets, and two callout-box styles ("key idea" / dark, "think about it" / light)
for the one-sentence takeaway of a slide. Colors are our own choice (navy/blue), not copied
pixel-for-pixel from the reference.

Run: uv run python -m src.presentation.build_slides
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "paper", "figures")
OUT_PATH = os.path.join(PROJECT_ROOT, "paper", "presentation", "slides.pptx")

# --- Palette -----------------------------------------------------------------
NAVY = RGBColor(0x1B, 0x2A, 0x5E)       # header bars / primary
NAVY_DARK = RGBColor(0x11, 0x1C, 0x40)  # callout "key idea" fill
STEEL = RGBColor(0x4A, 0x63, 0xA8)      # secondary accent
INK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x5A, 0x5A, 0x5A)
FAINT = RGBColor(0xC7, 0xCC, 0xDD)      # inactive agenda items
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
THINK_BG = RGBColor(0xFC, 0xEF, 0xE3)   # "think about it" light callout
THINK_INK = RGBColor(0x5A, 0x33, 0x0B)
# Official per-class colors, matching every figure in the paper (src/colmap_io/semantic_voting.py)
CLASS_COLORS = {
    "background": RGBColor(0x80, 0x80, 0x80),
    "deck": RGBColor(0xFF, 0x00, 0x00),
    "stay_cable": RGBColor(0x00, 0xC8, 0xC8),
    "tower": RGBColor(0x00, 0xB4, 0x00),
    "foundation": RGBColor(0xD2, 0xB4, 0x00),
}

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.45)
CONTENT_TOP = Inches(1.05)

AGENDA = [
    "Part 1 — The task",
    "Part 2 — Related work",
    "Part 3 — The backbone",
    "Part 4 — Our adaptations",
    "Part 5 — Experiments and results",
    "Part 6 — Discussion and conclusion",
]

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]

_slide_no = [0]


def new_slide():
    return prs.slides.add_slide(BLANK)


def set_bg(slide, color=WHITE):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def _no_line(shape):
    shape.line.fill.background()


def rect(slide, x, y, w, h, color, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if line is None:
        _no_line(shp)
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    return shp


def rounded(slide, x, y, w, h, color):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    _no_line(shp)
    shp.shadow.inherit = False
    try:
        shp.adjustments[0] = 0.06
    except IndexError:
        pass
    return shp


def oval(slide, x, y, d, color):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, d, d)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    _no_line(shp)
    shp.shadow.inherit = False
    return shp


def textbox(slide, x, y, w, h, text, size=16, color=INK, bold=False, italic=False,
            align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Calibri", line_spacing=1.0,
            wrap=True):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = font
        r.font.color.rgb = color
    return tb


_page_number_runs = []  # (run, slide_index) pairs, fixed up to the real total at save time


def header(slide, title, subtitle=None):
    _slide_no[0] += 1
    rect(slide, 0, 0, SLIDE_W, Inches(0.78), NAVY)
    textbox(slide, MARGIN, Inches(0.09), Inches(11.5), Inches(0.6), title,
            size=24, color=WHITE, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    if subtitle:
        textbox(slide, MARGIN, Inches(0.82), Inches(12), Inches(0.35), subtitle,
                size=13, color=GRAY, italic=True)
    pg = textbox(slide, SLIDE_W - Inches(1.1), SLIDE_H - Inches(0.4), Inches(0.7), Inches(0.3),
                 f"{_slide_no[0]} / ?", size=10, color=GRAY, align=PP_ALIGN.RIGHT)
    _page_number_runs.append((pg.text_frame.paragraphs[0].runs[0], _slide_no[0]))


def source(slide, text):
    textbox(slide, MARGIN, SLIDE_H - Inches(0.42), Inches(11), Inches(0.32),
            text, size=10, color=GRAY, italic=True)


def numbered_bullet(slide, n, head, body, x, y, w, num_color=NAVY, head_size=17, body_size=13.5):
    d = Inches(0.34)
    oval(slide, x, y + Inches(0.02), d, num_color)
    textbox(slide, x, y + Inches(0.02), d, d, str(n), size=13, color=WHITE, bold=True,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    textbox(slide, x + Inches(0.5), y, w - Inches(0.5), Inches(0.4), head, size=head_size,
            bold=True, color=INK)
    if body:
        textbox(slide, x + Inches(0.5), y + Inches(0.38), w - Inches(0.5), Inches(0.7), body,
                size=body_size, color=GRAY, line_spacing=1.05)


def dot_bullet(slide, head, body, x, y, w, color=STEEL, head_size=16, body_size=13):
    d = Inches(0.13)
    oval(slide, x, y + Inches(0.09), d, color)
    textbox(slide, x + Inches(0.28), y, w - Inches(0.28), Inches(0.32), head, size=head_size,
            bold=True, color=INK)
    if body:
        textbox(slide, x + Inches(0.28), y + Inches(0.34), w - Inches(0.28), Inches(0.7), body,
                size=body_size, color=GRAY, line_spacing=1.05)


def callout(slide, text, x, y, w, h, kind="key"):
    fill = NAVY_DARK if kind == "key" else THINK_BG
    ink = WHITE if kind == "key" else THINK_INK
    box = rounded(slide, x, y, w, h, fill)
    label = "Key idea:  " if kind == "key" else "Think about it:  "
    tb = box.text_frame
    tb.word_wrap = True
    tb.vertical_anchor = MSO_ANCHOR.MIDDLE
    tb.margin_left = Inches(0.22)
    tb.margin_right = Inches(0.22)
    tb.margin_top = Inches(0.08)
    tb.margin_bottom = Inches(0.08)
    p = tb.paragraphs[0]
    r1 = p.add_run()
    r1.text = label
    r1.font.bold = True
    r1.font.size = Pt(14)
    r1.font.color.rgb = ink
    r2 = p.add_run()
    r2.text = text
    r2.font.size = Pt(14)
    r2.font.color.rgb = ink
    return box


def picture(slide, relpath, x, y, max_w, max_h):
    from PIL import Image
    path = os.path.join(PROJECT_ROOT, relpath)
    with Image.open(path) as im:
        ar = im.width / im.height
    w, h = max_w, max_w / ar
    if h > max_h:
        h = max_h
        w = max_h * ar
    slide.shapes.add_picture(path, x + (max_w - w) // 2, y + (max_h - h) // 2, width=w, height=h)


def table(slide, headers, rows, x, y, w, h, col_widths=None, header_fill=NAVY,
          bold_row=None, font_size=13):
    n_rows, n_cols = len(rows) + 1, len(headers)
    gt = slide.shapes.add_table(n_rows, n_cols, x, y, w, h).table
    if col_widths:
        for i, cw in enumerate(col_widths):
            gt.columns[i].width = cw
    for j, htext in enumerate(headers):
        cell = gt.cell(0, j)
        cell.text = htext
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(font_size)
                r.font.color.rgb = WHITE
    for i, row in enumerate(rows):
        is_bold = bold_row is not None and i == bold_row
        for j, val in enumerate(row):
            cell = gt.cell(i + 1, j)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xF2, 0xF3, 0xF8) if i % 2 == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
                for r in p.runs:
                    r.font.size = Pt(font_size)
                    r.font.bold = is_bold
                    r.font.color.rgb = NAVY if is_bold else INK
    return gt


def speaker_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def divider(part_idx, title):
    s = new_slide()
    set_bg(s)
    header(s, title)
    for i, item in enumerate(AGENDA):
        active = i == part_idx
        y = Inches(1.5) + Inches(0.85) * i
        d = Inches(0.4)
        oval(s, Inches(0.9), y, d, NAVY if active else FAINT)
        textbox(s, Inches(0.9), y, d, d, str(i + 1), size=15, color=WHITE, bold=True,
                align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        textbox(s, Inches(1.55), y - Inches(0.03), Inches(9), Inches(0.5), item,
                size=22, bold=active, color=NAVY if active else FAINT,
                anchor=MSO_ANCHOR.MIDDLE)
    return s


# =============================================================================
# 1. Title
# =============================================================================
s = new_slide()
set_bg(s)
rounded(s, Inches(0.9), Inches(1.6), Inches(11.5), Inches(2.2), NAVY)
textbox(s, Inches(1.2), Inches(1.85), Inches(10.9), Inches(1.0),
        "Multi-View Semantic 3D Gaussian Splatting for\nComponent-Aware Bridge Reconstruction from UAV Imagery",
        size=26, bold=True, color=WHITE, align=PP_ALIGN.CENTER, line_spacing=1.1)
textbox(s, Inches(1.2), Inches(3.05), Inches(10.9), Inches(0.6),
        "Rendering a bridge's appearance and its structural parts from any camera pose, in one pass",
        size=15, italic=True, color=RGBColor(0xD8, 0xDC, 0xF0), align=PP_ALIGN.CENTER)
textbox(s, Inches(1.2), Inches(4.3), Inches(10.9), Inches(0.5),
        "Quoc-Bao Ta      Trung Ha Nguyen      Thai Ha Dang",
        size=18, color=INK, align=PP_ALIGN.CENTER)
textbox(s, Inches(1.2), Inches(4.85), Inches(10.9), Inches(0.4),
        "IC-SHM 2026 — Project 2: Multi-view Semantic 3D Reconstruction of Bridge Structures",
        size=14, color=GRAY, align=PP_ALIGN.CENTER)
textbox(s, Inches(1.2), Inches(5.25), Inches(10.9), Inches(0.4),
        "[Team identifier]", size=14, color=GRAY, align=PP_ALIGN.CENTER)
textbox(s, Inches(1.2), Inches(6.3), Inches(10.9), Inches(0.4),
        "September 2026", size=13, color=GRAY, align=PP_ALIGN.CENTER)
speaker_notes(s, "Title slide. Replace [Team identifier] with the registered team ID before "
                 "recording. Source: paper/DRAFT.md title block.")

# =============================================================================
# 2. The story in one page
# =============================================================================
s = new_slide()
set_bg(s)
header(s, "The story in one page")
story = [
    ("The task", "Given 400 posed UAV images of a suspension bridge, render both an RGB "
                  "image and a 5-class semantic map from an arbitrary camera pose."),
    ("What existed before", "3D Gaussian Splatting renders photorealistic RGB but carries no "
                             "semantics; 2D segmentation gives labels but not a renderable 3D scene."),
    ("Our backbone", "Semantic 3D Gaussian Splatting: give each Gaussian 5 extra semantic "
                      "logits and render RGB + semantics in one fused rasterization pass."),
    ("Our additions", "Multi-view plurality-vote semantic warm-start; SegFormer pseudo-labels "
                       "for the 100 unlabeled images; three real bugs found and fixed."),
    ("The experiment", "PSNR 22.43 dB, SSIM 0.854, LPIPS 0.321, structural mIoU 92.08% "
                        "on 30 held-out test views."),
    ("What we learned", "High cable IoU reflects a coarse annotation convention, not precise "
                         "cable geometry recovery — and after fixing other bugs, resolution "
                         "matters less than it first appeared to."),
]
for i, (h, b) in enumerate(story):
    numbered_bullet(s, i + 1, h, b, MARGIN, Inches(1.15) + Inches(1.0) * i, Inches(12.3))
speaker_notes(s, "One-page summary of the whole talk, in the same spirit as a plain-language "
                 "abstract. Source: paper/DRAFT.md Abstract and Section 1 (Introduction).")

# =============================================================================
# Part 1 -- The task
# =============================================================================
divider(0, "Part 1: The task")
speaker_notes(prs.slides[-1], "We start with what the contest actually asks us to build.")

# 4. Task & data
s = new_slide()
set_bg(s)
header(s, "What exactly are we asked to build?")
textbox(s, MARGIN, Inches(1.1), Inches(3.6), Inches(0.4), "Input", size=18, bold=True, color=NAVY)
dot_bullet(s, "400 posed UAV images", "300 carry manual polygon labels; 100 do not",
           MARGIN, Inches(1.55), Inches(3.6))
dot_bullet(s, "Camera calibration + poses", "SfM estimates, documented as "
           "“reference only”", MARGIN, Inches(2.35), Inches(3.6))
textbox(s, MARGIN, Inches(3.35), Inches(3.6), Inches(0.4), "Required output", size=18, bold=True,
        color=NAVY)
dot_bullet(s, "RGB image", "matching a held-out photograph", MARGIN, Inches(3.8), Inches(3.6),
           color=CLASS_COLORS["deck"])
dot_bullet(s, "Semantic map", "5 classes, see center column", MARGIN, Inches(4.45), Inches(3.6),
           color=CLASS_COLORS["stay_cable"])
dot_bullet(s, "...from any query pose", "poses never captured during the flight",
           MARGIN, Inches(5.1), Inches(3.6))

textbox(s, Inches(4.6), Inches(1.1), Inches(3.6), Inches(0.4), "5 semantic classes", size=18,
        bold=True, color=NAVY)
classes = ["background", "deck", "stay_cable", "tower", "foundation"]
for i, c in enumerate(classes):
    y = Inches(1.6) + Inches(0.52) * i
    rect(s, Inches(4.6), y + Inches(0.03), Inches(0.32), Inches(0.32), CLASS_COLORS[c])
    textbox(s, Inches(5.05), y, Inches(3), Inches(0.4), c, size=15, color=INK,
            anchor=MSO_ANCHOR.MIDDLE)
callout(s, "The competition brief calls the physical component “main cable” (p. 9); "
           "the dataset's own class ID is stay_cable. We keep the dataset ID for scoring and "
           "name the physical part correctly in prose.", Inches(4.6), Inches(4.3), Inches(3.6),
        Inches(1.5), kind="think")

textbox(s, Inches(8.55), Inches(1.1), Inches(4.3), Inches(0.4), "Blind evaluation", size=18,
        bold=True, color=NAVY)
dot_bullet(s, "Visual fidelity", "PSNR, SSIM, LPIPS", Inches(8.55), Inches(1.6), Inches(4.3))
dot_bullet(s, "Semantic accuracy", "mIoU over the 4 structural classes", Inches(8.55),
           Inches(2.3), Inches(4.3))
dot_bullet(s, "Accuracy Score", "= 0.5 × Visual Fidelity + 0.5 × Semantic mIoU",
           Inches(8.55), Inches(3.0), Inches(4.3))
callout(s, "Our numbers are all measured on a local 30-view holdout carved from the released "
           "data — not the organizers' blind test set.", Inches(8.55), Inches(4.3),
        Inches(4.3), Inches(1.5), kind="key")
source(s, "Source: paper/DRAFT.md Section 1 (Introduction), Section 3.1; competition brief p. 9.")
speaker_notes(s, "The contest gives us 400 posed images and asks for a model that renders both "
                 "RGB and a semantic map from any camera pose, scored on a blind held-out set "
                 "we never see.")

# 5. Why hard
s = new_slide()
set_bg(s)
header(s, "Why is this hard? Three reasons")
reasons = [
    ("Coarse cable annotation", "Physical cable strands span only a few pixels, but the "
     "stay_cable polygon encloses the whole cable-and-hanger region — including sky and "
     "river between the strands. High overlap with this label does not mean precise strand "
     "geometry."),
    ("A quarter of the images are unlabeled", "100 of 400 images carry no manual annotation. "
     "Using their viewpoints for semantic supervision requires pseudo-labeling."),
    ("Camera poses are estimates, not ground truth", "Poses come from Structure-from-Motion "
     "and are documented as “reference only”, so residual geometric inconsistency can "
     "affect rendering quality."),
]
for i, (h, b) in enumerate(reasons):
    numbered_bullet(s, i + 1, h, b, MARGIN, Inches(1.2) + Inches(1.5) * i, Inches(12.3),
                     head_size=18, body_size=14)
callout(s, "None of these three problems is solved by a bigger model — each needs its own "
           "targeted fix, covered in Part 4.", MARGIN, Inches(6.15), Inches(12.3), Inches(0.9),
        kind="think")
source(s, "Source: paper/DRAFT.md Section 1 (Introduction).")
speaker_notes(s, "Three specific properties of this dataset make the task non-trivial, "
                 "independent of model capacity.")

# 6. Scoring formula
s = new_slide()
set_bg(s)
header(s, "How is the answer scored?")
textbox(s, MARGIN, Inches(1.2), Inches(12.3), Inches(0.5),
        "Accuracy Score  =  0.5 × Visual Fidelity (PSNR / SSIM / LPIPS)  +  0.5 × Semantic mIoU",
        size=20, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
dot_bullet(s, "PSNR / SSIM", "pixel and structural similarity between rendered and real RGB",
           MARGIN, Inches(2.2), Inches(12))
dot_bullet(s, "LPIPS", "a learned perceptual distance (AlexNet backbone), lower is better",
           MARGIN, Inches(2.8), Inches(12))
dot_bullet(s, "Structural mIoU", "mean IoU over the 4 structural classes; we also report the "
           "5-class mIoU (including background) for completeness", MARGIN, Inches(3.4),
           Inches(12))
callout(s, "The brief specifies the 0.5/0.5 weighting but does not define how PSNR/SSIM/LPIPS "
           "combine into one “Visual Fidelity” number. We document our own combination "
           "explicitly as illustrative, not an organizer-defined formula.", MARGIN, Inches(4.3),
        Inches(12.3), Inches(1.3), kind="think")
source(s, "Source: paper/DRAFT.md Section 4.3 (Metrics), Illustrative combined score.")
speaker_notes(s, "We are careful to separate the organizer's actual scoring rule from our own "
                 "documented choice for how to combine PSNR/SSIM/LPIPS internally.")

# =============================================================================
# Part 2 -- Related work
# =============================================================================
divider(1, "Part 2: What existed before")

# 8. NeRF -> 3DGS -> semantic variants
s = new_slide()
set_bg(s)
header(s, "From implicit fields to explicit, labelable primitives")
methods = [
    ("NeRF", "Implicit, coordinate-based network; renders via ray-marching. High quality, "
             "slow, and no natural place to attach a discrete label."),
    ("3D Gaussian Splatting (Kerbl et al.)", "Explicit anisotropic Gaussians, tile-based "
             "rasterization. Real-time, and every pixel traces to an addressable set of "
             "primitives — the property we build on."),
    ("Semantic-NeRF / Feature-3DGS / LangSplat / Gaussian Grouping", "Attach extra attributes "
             "(semantic logits, distilled 2D features, language embeddings, SAM-derived "
             "identity codes) to a renderable representation — but for different "
             "supervision and different downstream tasks than ours."),
]
for i, (h, b) in enumerate(methods):
    numbered_bullet(s, i + 1, h, b, MARGIN, Inches(1.2) + Inches(1.55) * i, Inches(12.3),
                     head_size=17, body_size=13.5)
callout(s, "Our task needs class IDs for a fixed set of bridge components — so we optimize "
           "five class logits directly against real and pseudo-labeled masks, rather than "
           "distilling an open-vocabulary feature field.", MARGIN, Inches(6.0), Inches(12.3),
        Inches(1.0), kind="key")
source(s, "Source: paper/DRAFT.md Section 2 (Related Work).")
speaker_notes(s, "We situate our method between plain 3DGS and the family of "
                 "attribute-augmented Gaussian methods.")

# 9. comparison table
s = new_slide()
set_bg(s)
header(s, "Where our design sits")
table(s, ["Method", "Renders RGB", "Native semantics", "Single fused pass", "Fixed class set"],
      [
          ["NeRF [6]", "yes", "no", "—", "—"],
          ["3D Gaussian Splatting [7]", "yes", "no", "—", "—"],
          ["Semantic-NeRF [8]", "yes", "yes", "no (ray integration)", "yes"],
          ["Feature-3DGS [9] / LangSplat [12]", "yes", "distilled features", "yes", "no (open-vocab)"],
          ["Ours", "yes", "yes", "yes", "yes"],
      ], MARGIN, Inches(1.3), Inches(12.3), Inches(3.2),
      col_widths=[Inches(3.4), Inches(2.1), Inches(2.6), Inches(2.4), Inches(1.8)])
dot_bullet(s, "Task A (2D pseudo-labeling) uses SegFormer",
           "a lightweight hierarchical-Transformer encoder (MiT-B0) with an all-MLP decoder — "
           "compact enough to fine-tune on a single GPU", MARGIN, Inches(4.9), Inches(12.3))
source(s, "Source: paper/DRAFT.md Section 2 (Related Work); reference numbers match the paper's own list.")
speaker_notes(s, "This table is our own summary, not copied from any single cited paper.")

# =============================================================================
# Part 3 -- The backbone
# =============================================================================
divider(2, "Part 3: The backbone")

# 11. 3DGS analogy
s = new_slide()
set_bg(s)
header(s, "What is a 3D Gaussian, intuitively?")
dot_bullet(s, "A soft, blurry ellipsoid in 3D space", "not a point, not a triangle — a "
           "smooth splat of color with a position, size, orientation, and opacity",
           MARGIN, Inches(1.2), Inches(6))
dot_bullet(s, "A scene = thousands of overlapping splats", "rendering a view means projecting "
           "every splat onto the image plane and blending them by depth",
           MARGIN, Inches(2.0), Inches(6))
dot_bullet(s, "“Alpha-compositing” = blending front-to-back", "each pixel's color is a "
           "weighted sum of every splat that covers it, nearest splats contributing most",
           MARGIN, Inches(2.8), Inches(6))
picture(s, "paper/figures/fig4_gaussian_architecture.png", Inches(6.9), Inches(1.1),
        Inches(6.0), Inches(2.9))
callout(s, "Because every rendered pixel traces back to an explicit, addressable set of "
           "Gaussians, it is natural to give each Gaussian one more attribute beyond color.",
        MARGIN, Inches(4.0), Inches(12.3), Inches(1.0), kind="key")
source(s, "Source: paper/DRAFT.md Section 3.4 (Task B: Semantic 3D Gaussian Splatting), Representation.")
speaker_notes(s, "A plain-language framing of 3D Gaussian Splatting before we add semantics to it.")

# 12. adding semantic channel
s = new_slide()
set_bg(s)
header(s, "Adding a semantic channel: one tensor, one rasterization pass")
dot_bullet(s, "Each Gaussian already carries", "position, scale, rotation, opacity, and RGB "
           "color (3 numbers, constrained to [0,1] by a sigmoid)", MARGIN, Inches(1.15), Inches(12.3))
dot_bullet(s, "We add 5 more numbers per Gaussian", "unconstrained semantic logits, one per "
           "class (background, deck, stay_cable, tower, foundation)", MARGIN, Inches(1.95), Inches(12.3))
dot_bullet(s, "Concatenate: 3 + 5 = 8 channels", "rasterized together in a single gsplat pass, "
           "sharing the exact same projection, depth order, and blending weights as color",
           MARGIN, Inches(2.75), Inches(12.3))
dot_bullet(s, "Split the output back apart", "channels 0–2 → RGB image; channels 3–7 → "
           "semantic logits → argmax → class map", MARGIN, Inches(3.55), Inches(12.3))
callout(s, "Because RGB and semantics share the same compositing weights, the two outputs are "
           "pixel-aligned by construction — no separate alignment step, no extra rasterization cost.",
        MARGIN, Inches(4.7), Inches(12.3), Inches(1.0), kind="key")
source(s, "Source: paper/DRAFT.md Section 3.4, Fused rendering.")
speaker_notes(s, "This is the core architectural idea: one shared rasterization pass, "
                 "two outputs split from the same 8-channel tensor.")

# 13. Figure 4 diagram
s = new_slide()
set_bg(s)
header(s, "The full picture: representation, rendering, and losses")
picture(s, "paper/figures/fig4_gaussian_architecture.png", Inches(0.6), Inches(1.0),
        Inches(12.1), Inches(6.1))
source(s, "Source: paper/DRAFT.md Figure 4 (Section 3.4).")
speaker_notes(s, "This diagram is reused verbatim from the paper. Walk through: geometry+opacity, "
                 "RGB (sigmoid), semantic logits (unconstrained) concatenate, one rasterization "
                 "pass, argmax for the class map, and the two losses that update everything.")

# =============================================================================
# Part 4 -- Our adaptations
# =============================================================================
divider(3, "Part 4: Our adaptations")

# 15. sparse init + semantic warm-start
s = new_slide()
set_bg(s)
header(s, "Starting from a good guess, not from noise")
dot_bullet(s, "Triangulate a sparse point cloud", "86,319 tracks from the 370 training-pool "
           "images (test images excluded, see Part 5) → 84,086 triangulated → 82,518 after "
           "outlier filtering", MARGIN, Inches(1.15), Inches(12.3))
dot_bullet(s, "Position + color initialize each Gaussian", "not random — real triangulated "
           "position, real averaged pixel color from every observing view", MARGIN, Inches(2.15),
           Inches(12.3))
dot_bullet(s, "Semantic logits initialize by multi-view plurality vote", "every labeled view "
           "observing a point votes for the class at its feature-track pixel; the winning class "
           "gets logit +2, others -2 (a confident but not saturated prior)", MARGIN, Inches(3.15),
           Inches(12.3))
picture(s, "paper/figures/fig3_cable_voting.png", Inches(1.5), Inches(4.35), Inches(10.3),
        Inches(2.6))
source(s, "Source: paper/DRAFT.md Section 3.2, Section 3.4 (Initialization); Figure 3.")
speaker_notes(s, "Both geometry and semantics start from a real, data-derived guess, "
                 "which the training loss then refines.")

# 16. pseudo-labeling
s = new_slide()
set_bg(s)
header(s, "Task A: turning 100 unlabeled images into training signal")
picture(s, "paper/figures/fig2_segformer_architecture.png", Inches(0.6), Inches(1.1),
        Inches(7.4), Inches(3.0))
dot_bullet(s, "Fine-tune SegFormer (MiT-B0)", "on 240 labeled images, 80 epochs",
           Inches(8.3), Inches(1.2), Inches(4.6))
dot_bullet(s, "Select checkpoint by validation mIoU", "on a separate 30-image internal "
           "validation split — never the 30-image test split", Inches(8.3), Inches(2.0),
           Inches(4.6))
dot_bullet(s, "Best validation mIoU: 81.67%", "at epoch 69", Inches(8.3), Inches(2.9),
           Inches(4.6))
dot_bullet(s, "Predict pseudo-masks for 100 images", "270 labeled + 100 "
           "pseudo-labeled = 370 views supervise Task B", Inches(8.3), Inches(3.7), Inches(4.6))
callout(s, "Checkpoint selection never touches the 30-image test split — an earlier version "
           "of our pipeline leaked this and we fixed it (next slides).", MARGIN, Inches(4.6),
        Inches(12.3), Inches(1.1), kind="think")
source(s, "Source: paper/DRAFT.md Section 3.3 (Task A), Figure 2.")
speaker_notes(s, "SegFormer's job is only to extend semantic supervision to the unlabeled "
                 "images; it never contributes to the final Gaussian model directly.")

# 17. Adaptation 1: mask alignment
s = new_slide()
set_bg(s)
header(s, "Adaptation 1: align masks to the space we actually render in")
textbox(s, MARGIN, Inches(1.15), Inches(2.0), Inches(0.4), "Problem:", size=16, bold=True, color=NAVY)
textbox(s, Inches(2.3), Inches(1.15), Inches(10.4), Inches(0.9),
        "Ground-truth and pseudo masks are rasterized/predicted on the original, "
        "lens-distorted photos. But Gaussian Splatting assumes an ideal pinhole camera, so "
        "training and rendering happen in an undistorted image space.",
        size=15, color=INK, line_spacing=1.1)
textbox(s, MARGIN, Inches(2.35), Inches(2.0), Inches(0.4), "Consequence:", size=16, bold=True,
        color=NAVY)
textbox(s, Inches(2.3), Inches(2.35), Inches(10.4), Inches(0.8),
        "A mask compared pixel-for-pixel against an undistorted render is silently "
        "misaligned by several pixels near the frame edges — worst exactly where thin "
        "classes like the cable live.", size=15, color=INK, line_spacing=1.1)
textbox(s, MARGIN, Inches(3.45), Inches(2.0), Inches(0.4), "Fix:", size=16, bold=True, color=NAVY)
textbox(s, Inches(2.3), Inches(3.45), Inches(10.4), Inches(0.8),
        "Undistort every ground-truth and pseudo mask with the same per-pixel remap used for "
        "images, with nearest-neighbor interpolation so discrete class IDs never blur.",
        size=15, color=INK, line_spacing=1.1)
callout(s, "A geometric consistency bug, not a modeling choice — fixed once, at the data "
           "layer, so every downstream number is computed on correctly aligned data.",
        MARGIN, Inches(4.5), Inches(12.3), Inches(1.0), kind="key")
source(s, "Source: paper/DRAFT.md Section 3.2 (Camera Geometry and Sparse Point Initialization).")
speaker_notes(s, "This was the first of several concrete pipeline bugs we found and fixed "
                 "during development, each verified against real re-measured numbers.")

# 18. Adaptation 2: real color init
s = new_slide()
set_bg(s)
header(s, "Adaptation 2: initialize Gaussian color from real pixels")
textbox(s, MARGIN, Inches(1.1), Inches(2.0), Inches(0.4), "Problem:", size=16, bold=True, color=NAVY)
textbox(s, Inches(2.3), Inches(1.1), Inches(10.4), Inches(0.8),
        "An earlier version of our pipeline initialized each Gaussian's RGB from a fixed "
        "5-color class-legend swatch (the same palette used for figures) — not the point's "
        "actual observed appearance.", size=15, color=INK, line_spacing=1.1)
textbox(s, MARGIN, Inches(2.15), Inches(2.0), Inches(0.4), "Hypothesis:", size=16, bold=True,
        color=NAVY)
textbox(s, Inches(2.3), Inches(2.15), Inches(10.4), Inches(0.8),
        "Starting the photometric loss from a class-swatch color — not real appearance — "
        "makes optimization spend part of its fixed 40,000-step budget just correcting the "
        "starting point, converging to a worse final fit.", size=15, color=INK, line_spacing=1.1)
textbox(s, MARGIN, Inches(3.2), Inches(2.0), Inches(0.4), "Fix:", size=16, bold=True, color=NAVY)
textbox(s, Inches(2.3), Inches(3.2), Inches(10.4), Inches(0.6),
        "Sample each point's real RGB from every observing image, averaged over views.",
        size=15, color=INK)
table(s, ["Metric", "Before (class-swatch color)", "After (real pixel color)"],
      [["PSNR", "21.83 dB", "22.13 dB"], ["Structural mIoU", "88.97%", "91.04%"]],
      MARGIN, Inches(4.0), Inches(9.0), Inches(1.3),
      col_widths=[Inches(2.5), Inches(3.25), Inches(3.25)])
callout(s, "Test of the hypothesis: every single metric improved after the fix.",
        Inches(9.7), Inches(4.0), Inches(2.9), Inches(1.3), kind="key")
source(s, "Source: paper/DRAFTING_NOTES.md, external review fix #4; src/colmap_io/reconstructor.py sample_point_colors.")
speaker_notes(s, "A real bug with a real before/after measurement — not a hyperparameter "
                 "tweak. Numbers come from re-running the full pipeline both ways.")

# 19. Adaptation 3: test-leak fix
s = new_slide()
set_bg(s)
header(s, "Adaptation 3: keep the test split out of the sparse cloud too")
textbox(s, MARGIN, Inches(1.1), Inches(2.0), Inches(0.4), "Problem:", size=16, bold=True, color=NAVY)
textbox(s, Inches(2.3), Inches(1.1), Inches(10.4), Inches(0.9),
        "Our training loss and semantic vote always excluded the 30 test images — but sparse "
        "triangulation and Gaussian color sampling used observations from all 400 images, "
        "test images included.", size=15, color=INK, line_spacing=1.1)
textbox(s, MARGIN, Inches(2.3), Inches(2.0), Inches(0.4), "Fix:", size=16, bold=True, color=NAVY)
textbox(s, Inches(2.3), Inches(2.3), Inches(10.4), Inches(0.9),
        "Exclude the 30 test images' 2D observations from every feature track before "
        "triangulation; their poses still load (needed to render them at evaluation time), "
        "but they cannot influence any point's position or color.", size=15, color=INK,
        line_spacing=1.1)
table(s, ["Metric", "Before (test leaked in)", "After (test fully excluded)"],
      [["PSNR", "22.13 dB", "22.43 dB"], ["Structural mIoU", "91.04%", "92.08%"]],
      MARGIN, Inches(3.5), Inches(9.0), Inches(1.3),
      col_widths=[Inches(2.5), Inches(3.25), Inches(3.25)])
callout(s, "Again, every metric improved — consistent with the dense flight's >99% "
           "frame-to-frame overlap making a training-adjacent frame almost as informative as "
           "the excluded test frame itself.", MARGIN, Inches(5.0), Inches(12.3), Inches(1.1),
        kind="key")
source(s, "Source: paper/DRAFTING_NOTES.md, external review fix #5; src/colmap_io/reconstructor.py exclude_image_names.")
speaker_notes(s, "This is the most subtle of the three fixes — the leak was only in the "
                 "one-time geometric/photometric initialization, not the training loop.")

# 20. Pipeline overview
s = new_slide()
set_bg(s)
header(s, "The whole pipeline, end to end")
picture(s, "paper/figures/fig1_pipeline.png", Inches(0.7), Inches(1.0), Inches(12.0), Inches(6.1))
source(s, "Source: paper/DRAFT.md Figure 1 (Section 3).")
speaker_notes(s, "This is the actual pipeline diagram from the paper, with the real current "
                 "counts: 82,518 sparse points, 370 supervised views, 600,583 final Gaussians.")

# =============================================================================
# Part 5 -- Experiments and results
# =============================================================================
divider(4, "Part 5: Experiments and results")

# 22. setup
s = new_slide()
set_bg(s)
header(s, "Experimental setup")
rows = [
    ("Data", "400 UAV images, 1320×989, shared SIMPLE_RADIAL calibration "
             "(f≈925.7 px, k₁≈0.009); 300 labeled, 100 unlabeled"),
    ("Split", "240 / 30 / 30 labeled (train / Task A validation / test), strided by flight "
              "order; +100 unlabeled via pseudo-labels"),
    ("Metrics", "PSNR, SSIM, LPIPS (visual fidelity); per-class and structural mIoU "
                "(semantic accuracy), pooled over all test pixels"),
    ("Compute", "Single NVIDIA RTX 3080 (10 GB); 40,000 iterations, seed 42, "
                "≈46 min full resolution"),
    ("Fair comparison", "Both resolution-ablation runs share the same split, initialization, "
                        "loss weights, and density-control schedule — only resolution differs"),
]
for i, (label, val) in enumerate(rows):
    y = Inches(1.2) + Inches(1.0) * i
    textbox(s, MARGIN, y, Inches(2.3), Inches(0.4), label, size=16, bold=True, color=NAVY,
            align=PP_ALIGN.RIGHT)
    textbox(s, Inches(3.0), y, Inches(9.8), Inches(0.9), val, size=14, color=INK,
            line_spacing=1.05)
source(s, "Source: paper/DRAFT.md Section 3.6 (Evaluation Protocol), Section 4.1–4.2.")
speaker_notes(s, "The setup slide before results, matching the paper's own experimental "
                 "protocol section.")

# 23. main results
s = new_slide()
set_bg(s)
header(s, "Main result: 30-view local test split")
table(s, ["Metric", "Value"],
      [["PSNR", "22.43 dB"], ["SSIM", "0.854"], ["LPIPS", "0.321"],
       ["Structural mIoU (4 classes)", "92.08%"], ["mIoU incl. background (5 classes)", "93.5%"],
       ["Illustrative Accuracy Score", "0.823"]],
      MARGIN, Inches(1.2), Inches(6.0), Inches(3.2), bold_row=3,
      col_widths=[Inches(3.6), Inches(2.4)])
table(s, ["Class", "IoU"],
      [["deck", "95.72%"], ["tower", "92.37%"], ["stay_cable", "91.76%"],
       ["foundation", "88.49%"], ["background", "99.20%"]],
      Inches(6.8), Inches(1.2), Inches(6.0), Inches(2.9),
      col_widths=[Inches(3.6), Inches(2.4)])
callout(s, "For reference: Task A's own 2D validation mIoU is 81.67% (30-image internal "
           "validation, not the test split above) — a different task, split, and role.",
        MARGIN, Inches(4.6), Inches(12.3), Inches(1.0), kind="think")
source(s, "Source: paper/DRAFT.md Tables 1–3 (Section 5.1).")
speaker_notes(s, "We keep Task A's own mIoU clearly separate from Task B's, since both are "
                 "called mIoU but measure different things.")

# 24. per-class discussion
s = new_slide()
set_bg(s)
header(s, "Per-class IoU: why is the thinnest class not the worst?")
picture(s, "paper/figures/fig5_per_class_iou.png", Inches(0.5), Inches(1.0), Inches(5.6),
        Inches(4.1))
dot_bullet(s, "Cable strands are physically the thinnest component", "yet stay_cable IoU "
           "(91.76%) is within a point of tower and clearly ahead of foundation", Inches(6.4),
           Inches(1.2), Inches(6.2))
dot_bullet(s, "The scored class is a coarse polygon, not the strands", "cable covers 7.12% of "
           "test pixels — more than deck's 4.99% — because the annotation encloses sky and "
           "river between strands", Inches(6.4), Inches(2.3), Inches(6.2))
dot_bullet(s, "High cable IoU measures annotation agreement",
           "not precise recovery of individual cable geometry", Inches(6.4), Inches(3.5),
           Inches(6.2))
dot_bullet(s, "Foundation's lower IoU is not simply fewer observations",
           "it receives more average observations per point than deck (6.57 vs. 3.12), so "
           "observation count alone does not explain the ranking", Inches(6.4), Inches(4.6),
           Inches(6.2))
callout(s, "We report these as descriptive statistics, not a confirmed causal explanation — "
           "isolating the true cause would need a dedicated error analysis we have not run.",
        MARGIN, Inches(5.9), Inches(12.3), Inches(1.0), kind="think")
source(s, "Source: paper/DRAFT.md Section 5.2 (Discussion — Per-Class Behavior).")
speaker_notes(s, "This slide deliberately keeps the paper's own hedged, non-causal framing "
                 "rather than overstating what the descriptive statistics show.")

# 25. qualitative
s = new_slide()
set_bg(s)
header(s, "Qualitative check: four held-out views")
picture(s, "paper/figures/fig6_qualitative_grid.png", Inches(0.9), Inches(1.0), Inches(11.5),
        Inches(5.6))
source(s, "Source: paper/DRAFT.md Figure 6 (Section 5.3): views 010, 050, 250, 300.")
speaker_notes(s, "View 250 is the cleanest RGB reconstruction by every metric; 010 and 300 "
                 "show comparable RGB degradation. Notably, the semantic map stays close to "
                 "ground truth in all four views even where RGB is visibly degraded.")

# 26. interpolation + splat viewer
s = new_slide()
set_bg(s)
header(s, "Beyond the flight line: interpolated poses and the full 3D structure")
picture(s, "paper/figures/fig7_interpolation.png", Inches(0.5), Inches(1.0), Inches(6.0),
        Inches(1.9))
textbox(s, Inches(0.5), Inches(3.0), Inches(6.0), Inches(0.8),
        "5 poses interpolated between two real test views (images 280, 300). Endpoints and "
        "intermediate frames all show some artifacts; broad component regions stay "
        "identifiable but boundaries fragment.", size=13, color=GRAY, line_spacing=1.1)
picture(s, "paper/figures/fig8_splat_render.png", Inches(6.8), Inches(1.0), Inches(6.0),
        Inches(1.9))
textbox(s, Inches(6.8), Inches(3.0), Inches(6.0), Inches(1.1),
        "Interactive-viewer screenshots of the same model: (a) learned RGB, (b) each Gaussian "
        "recolored by its own argmax class before rendering. Panel (b) is a different "
        "visualization from the evaluated metric, which argmaxes after compositing.",
        size=13, color=GRAY, line_spacing=1.1)
callout(s, "Rendering a pose is not the same as being accurate at that pose — we show these "
           "views to be transparent about where the system still struggles.", MARGIN,
        Inches(4.5), Inches(12.3), Inches(1.0), kind="think")
source(s, "Source: paper/DRAFT.md Figures 7–8 (Section 5.3).")
speaker_notes(s, "We are explicit that Figure 8 panel (b) is not the same computation as the "
                 "evaluated semantic output — it recolors before compositing, the metric "
                 "argmaxes after.")

# 27. resolution ablation
s = new_slide()
set_bg(s)
header(s, "Ablation: does training resolution matter?")
table(s, ["Training resolution", "PSNR", "SSIM", "LPIPS", "Structural mIoU"],
      [["Half (660×494)", "22.03", "0.838", "0.330", "89.71%"],
       ["Full (1320×989)", "22.43", "0.854", "0.321", "92.08%"]],
      MARGIN, Inches(1.3), Inches(11.5), Inches(1.5), bold_row=1,
      col_widths=[Inches(3.3), Inches(2.05), Inches(2.05), Inches(2.05), Inches(2.05)])
dot_bullet(s, "Full resolution wins on every metric", "+0.40 dB PSNR, +0.016 SSIM, "
           "−0.009 LPIPS, +2.37 points structural mIoU", MARGIN, Inches(3.2), Inches(12.3))
dot_bullet(s, "Cost: ≈46 minutes vs. ≈18 minutes on the same GPU", "a modest absolute "
           "difference for the same 40,000-step budget", MARGIN, Inches(3.9), Inches(12.3))
callout(s, "These results support full resolution for this dataset and budget; we have not "
           "quantified variation across random seeds, other scenes, or other training budgets.",
        MARGIN, Inches(4.9), Inches(12.3), Inches(1.0), kind="think")
source(s, "Source: paper/DRAFT.md Table 4, Section 5.5 (Ablation: Training Resolution).")
speaker_notes(s, "Full resolution is the better default within what we actually tested — "
                 "we do not overstate generalization beyond this paired experiment.")

# =============================================================================
# Part 6 -- Discussion and conclusion
# =============================================================================
divider(5, "Part 6: Discussion and conclusion")

# 29. Honest limitations
s = new_slide()
set_bg(s)
header(s, "Honest limitations")
lims = [
    ("The split tests interpolation, not extrapolation", "the strided test split places a "
     "training frame next to almost every held-out frame along a dense flight path"),
    ("Cable IoU reflects an annotation convention", "not verified precision at the level of "
     "individual strands"),
    ("Accuracy at distant, off-trajectory viewpoints is unverified", "we can render an "
     "arbitrary pose, but rendering a pose is not the same as being accurate there"),
    ("Not evaluated for the actual SHM end task", "no damage detection, deformation "
     "measurement, or structural response estimation has been tested"),
    ("Single run per configuration", "no multi-seed variance estimate for the headline numbers "
     "or the resolution ablation"),
]
for i, (h, b) in enumerate(lims):
    numbered_bullet(s, i + 1, h, b, MARGIN, Inches(1.15) + Inches(1.08) * i, Inches(12.3),
                     head_size=16, body_size=13)
source(s, "Source: paper/DRAFT.md Section 3.6, Section 5.3, Section 6 (Conclusion).")
speaker_notes(s, "A dedicated limitations slide, matching the paper's own explicit hedging "
                 "rather than a generic closing disclaimer.")

# 30. Conclusion
s = new_slide()
set_bg(s)
header(s, "Conclusion")
concl = [
    "A semantic 3D Gaussian Splatting system that renders aligned RGB and 5-class semantic "
    "maps from any query pose in one fused rasterization pass.",
    "Sparse geometry, observed colors, and multi-view plurality votes warm-start the "
    "representation instead of starting from noise.",
    "SegFormer pseudo-labels extend semantic supervision from 270 to 370 views at no extra "
    "annotation cost.",
    "On a local 30-view test split: PSNR 22.43 dB, SSIM 0.854, LPIPS 0.321, "
    "92.08% structural mIoU.",
    "Three real pipeline bugs were found and fixed during development — mask alignment, "
    "color initialization, and test-split leakage into triangulation — each verified by a "
    "measured before/after improvement.",
    "Results establish agreement with the released annotations near the flight trajectory, "
    "not survey-grade geometric accuracy or unrestricted novel-view fidelity.",
]
for i, t in enumerate(concl):
    numbered_bullet(s, i + 1, t, "", MARGIN, Inches(1.15) + Inches(0.85) * i, Inches(12.3),
                     head_size=15)
source(s, "Source: paper/DRAFT.md Section 6 (Conclusion).")
speaker_notes(s, "Closing summary, staying within exactly what the paper itself claims.")

# 31. References
s = new_slide()
set_bg(s)
header(s, "References and code")
refs = [
    "Kerbl, Kopanas, Leimkühler, Drettakis (2023) — 3D Gaussian Splatting for Real-Time "
    "Radiance Field Rendering. ACM ToG.",
    "Mildenhall et al. (2020) — NeRF: Representing Scenes as Neural Radiance Fields. ECCV.",
    "Zhi, Laidlow, Leutenegger, Davison (2021) — Semantic-NeRF. ICCV.",
    "Zhou et al. (2024) — Feature 3DGS. CVPR.  ·  Qin et al. (2024) — LangSplat. CVPR.",
    "Xie et al. (2021) — SegFormer. NeurIPS.  ·  Ye et al. (2025) — gsplat. JMLR.",
    "Lin, Abe, Zheng, Li, Chun (2025) — A structure-oriented loss function for bridge point "
    "clouds. CACAIE.",
    "Full reference list: paper/DRAFT.md References (19 numbered entries).",
]
for i, t in enumerate(refs):
    dot_bullet(s, t, "", MARGIN, Inches(1.15) + Inches(0.62) * i, Inches(12.3), head_size=14)
textbox(s, MARGIN, Inches(5.7), Inches(12.3), Inches(0.4),
        "Code: github.com/zeki-aitech/ic-shm-2026-project-2", size=15, bold=True, color=NAVY)
textbox(s, MARGIN, Inches(6.5), Inches(12.3), Inches(0.6), "Thank you — questions?", size=24,
        bold=True, color=NAVY, align=PP_ALIGN.CENTER)
source(s, "Source: paper/DRAFT.md References section and Section 1 (repository link).")
speaker_notes(s, "Closing slide with the repository link, matching Section 1's stated URL.")


total = _slide_no[0]
for run, n in _page_number_runs:
    run.text = f"{n} / {total}"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
prs.save(OUT_PATH)
print(f"[build_slides] wrote {OUT_PATH} ({total} slides)")
