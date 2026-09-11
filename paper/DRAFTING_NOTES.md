# Drafting Notes — IC-SHM 2026 Project 2 Paper

`DRAFT.md` is a structural/content draft of the paper (target: 10-15 pages, official IC-SHM
template, English) — it is NOT yet formatted in the official template. Pour its content into
the downloaded template once available from the contest website, before final submission.

This file tracks outstanding work and a record of what's already been done. Delete or archive
it once the paper is finalized and submitted — it is not part of the paper itself.

## Outstanding

The current submission preparation checklist is [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).
The manuscript uses 19 numbered references, including the competition brief as reference [4].

- [ ] **Future idea (optional, not required by anything in the paper as currently written):**
  ablate the semantic warm-start logit magnitude (`sem_init`, `src/gaussian_splatting/model.py`
  lines ~131/133 - currently a hardcoded `+2.0`/`-2.0`, chosen by feel, not by any ablation or
  formula on record). Came up when the user asked "why 2 and not 1.5/2.5/3" - answered informally
  in conversation (softmax(±m) gives P(voted class) = 65%/83%/93%/97%/99% for m=1/1.5/2/2.5/3,
  with gradient scale `P(1-P)` shrinking fast past m~2.5-3, so there's a broad, not sharply
  peaked, "reasonable zone" rather than one optimal value) but never verified empirically. If
  ever worth doing: train a few short (few-thousand-iteration) runs at m in {1, 2, 4} and compare
  either final mIoU or, more informatively, how much each is displaced from its own initial
  per-Gaussian class vote after training - would also double as indirect evidence for the
  "informed prior, not unbreakable label" claim Section 3.4 already makes about this design.
  Not blocking anything; only do this if there's spare time/compute and the user asks for it.
- [ ] Confirm exact page/formatting requirements once the official template is downloaded from
  the IC-SHM website and available locally.
- [ ] Reformat citations into the official template's required style once available
  (`DRAFT.md` References section) — if the template mandates author-year, both the in-text
  citations and the reference list's ordering will need to change accordingly.
- [ ] Team/author details: name(s), affiliation, IC-SHM 2026 Project 2 team identifier
  (`DRAFT.md` title block, currently `[NEEDS]`).
- [x] Repository URL included in Section 1. Public access and the exact submission commit
  still need verification during submission packaging.
- [ ] `docs/EXPERIMENT_PROGRESS_AND_FINDINGS.md` and `docs/SUBMISSION_CHECKLIST_AND_GUIDELINES.md`
  still reference the old strict-majority checkpoint's numbers/Gaussian count (602,363) - not
  updated in the plain-plurality baseline switch below (only `DRAFT.md` and `README.md` were).

## Done

- [x] **Retrained Table 4 (resolution ablation) under every current fix**, closing the item that
  had been open since fix #1 (it previously used checkpoints predating mask undistortion, the
  split fix, the color-init fix, and the triangulation test-leak fix - four independent stale
  axes at once). Full-resolution row reuses the exact checkpoint already reported as the main
  result (Table 2/3) rather than retraining a duplicate; only half-resolution needed a fresh run
  (`outputs/checkpoints/gaussians_halfres_test_excluded_init/`, `--downsample 0.5`, same
  40,000-iteration budget, ~18 min vs. ~46 min for full - both figures re-measured, not assumed).
  - **Half-resolution result**: PSNR 22.03 dB, SSIM 0.838, LPIPS 0.330, mIoU 89.71% (was
    21.79/0.831/0.355/85.77% pre-fix). Full-resolution (Table 2/3): PSNR 22.43 dB, SSIM 0.854,
    LPIPS 0.321, mIoU 92.08%.
  - **The resolution effect's direction held (full still beats half on every metric) but its size
    shrank substantially**: mIoU gap +2.37 points (was +5.7). An earlier version of this section
    had predicted retraining would leave "the direction and size... unaffected" - the direction
    claim held, the size claim didn't, so the text now reports the measured result and says so
    explicitly rather than quietly correcting the old prediction without comment.
  - Dropped the old aside comparing the new 40k-iteration half-res number against a much earlier,
    differently-configured 30k-iteration half-res run (87.96% mIoU) - that comparison already
    crossed too many independent axes before this fix, and now crosses five, so it no longer says
    anything meaningful; not replaced with anything, since re-running a 30k-iteration comparison
    wasn't asked for and isn't needed for this section's actual conclusion.
- [x] **Refreshed Figure 8 (splat viewer render, Section 5.3, optional) for the fix #5
  (test-split-clean triangulation) checkpoint.** Exported new `bridge_splat_test_excluded_init_
  {rgb,semantic}.ply` from the current canonical checkpoint via `export_ply.py`, user captured
  true-color and semantic screenshots in SuperSplat, composed via `compose_splat_screenshots.py`
  into `fig8_splat_render.png`. Deleted the superseded `bridge_splat_real_color_{rgb,semantic}.ply`
  exports from `outputs/renders/` (gitignored local artifacts).
- [x] **External review fix #5 (test-split leakage into triangulation/color-init) implemented
  and retrained.** The user's own review (echoing an external reviewer's finding) caught that
  `DRAFT.md` claimed the 30-image test split was "excluded from every stage of training," but
  `src/gaussian_splatting/train.py::prepare_training_data` actually triangulated the sparse point
  cloud and sampled Gaussian color init from all 400 images, test split included - only semantic
  voting and the training loss were genuinely train-only. Verified this in the code before
  acting (`PycolmapReconstructor(colmap_dir).load()` had no split awareness at all), then fixed
  it: added `exclude_image_names` to `PycolmapReconstructor`/`build_tracks`/`triangulate_tracks`
  (`src/colmap_io/reconstructor.py`) so excluded images' poses still load (needed to render them
  at eval time) but contribute no 2D observation to any feature track, so they cannot influence a
  triangulated point's position - and, since `sample_point_colors` only samples from a point's
  own observing images, color init is transitively excluded too, with no separate filtering
  needed there. `prepare_training_data` now computes the split before triangulating and passes
  the test filenames through. Added `tests/test_build_tracks_exclusion.py` (4 tests, synthetic
  fake reconstruction, no real pycolmap needed) covering the core exclusion logic.
  - **Verified against the real dataset before committing to a retrain**: 84,613 -> 82,518 points
    after excluding the 30 test images (86,319 tracks attempted, reprojection error essentially
    unchanged at 0.49px mean vs. the prior 0.50px) - consistent with the dense UAV trajectory's
    >99% frame-to-frame overlap making most of a test image's triangulation information
    redundant with its adjacent training frames.
  - **Retrained Task B once** (`outputs/checkpoints/gaussians/`, installed as canonical; prior
    checkpoint preserved as `gaussians_stale_test_leak_init_pre_20260911`) - 600,583 Gaussians
    (was 604,152), Gaussian count stabilizing at step 7,400 (unchanged) and loss plateauing by
    ~step 20,000 (unchanged, re-verified via moving-average check on the new log). Task A
    (SegFormer) needed no retraining - it never used `PycolmapReconstructor` at all.
  - **Result on the same 30-image test set**: every metric improved again - PSNR 22.43 dB (was
    22.13), SSIM 0.854 (was 0.853), LPIPS 0.321 (unchanged), mIoU 92.08% (was 91.04%; deck
    95.72%, stay_cable 91.76%, tower 92.37%, foundation 88.49%, background 99.20%), Accuracy
    Score 0.823 (was 0.816). Per-class ranking unchanged (deck > tower > cable > foundation) but
    deck's gap over the other three widened substantially, so Section 5.2's "cable lands within a
    point of deck and tower" framing was corrected to "within a point of tower" only (deck is now
    ~4 points ahead) - not just a number swap, a real framing fix. Also recomputed the
    per-class avg-observations-per-point statistic (foundation 6.57, deck 3.12 - was 4.66/2.68)
    since it depends on the now-different point cloud; the qualitative narrative (foundation gets
    *more* observing views than deck despite scoring lower) still holds.
  - **Re-verified Section 5.3's per-view qualitative claims against real per-view PSNR/SSIM/LPIPS**
    (not just eyeballing the new renders): the old checkpoint's "050 cleanest, 250 close behind"
    ordering no longer holds - on the new checkpoint 250 is actually the cleanest of the four
    views shown (010/050/250/300) and 050 second; 010 and 300 are now comparably weak rather than
    300 being unambiguously worst. Rewrote that paragraph with the real per-view numbers instead
    of re-describing the images by eye.
  - **Strengthened Section 3.6's evaluation-protocol paragraph and the Abstract** to explicitly
    state the test split is excluded from triangulation and color init too, not just training/
    validation/voting - this claim is now actually true, closing the gap between what the paper
    claimed and what the code did.
  - **Updated throughout `DRAFT.md`**: Abstract, Introduction (bullet 5), Section 3.2 (track/point
    counts, new exclusion description), Section 3.4/4.2 (point cloud and Gaussian counts, both
    occurrences), Section 3.6 (exclusion list), Table 2, Table 3, Table 4's footnote (now four
    independent axes of staleness, not three), Figure 5's surrounding text/caption, Section 5.2
    (deck/tower/cable/foundation discussion, rewritten where the framing itself changed, not just
    the numbers), Section 5.3 (per-view qualitative claims), Conclusion. Regenerated Figures 1
    (Gaussian/point counts), 4 (new view-300 render), 5 (per-class IoU chart), 6 (qualitative
    grid, re-rendered views 010/050/250/300), 7 (interpolation filmstrip, images 280/300), 10
    (Task B training curve). Figure 8 (splat viewer, optional) was not refreshed this round - it
    still needs a new SuperSplat capture from the current checkpoint, same as after fix #4.
    Updated `README.md`'s results table and pipeline description. Ran the full test suite
    (101 passed) after all changes.
- [x] **Added the Task A SegFormer architecture figure (new Figure 2, Section 3.3)**, produced by
  a Codex session working on this same repo in parallel with this one (per the user's earlier
  request to hand an architecture description to an external tool rather than have me draw it -
  the same pattern as the Figure 4/gaussian-architecture figure below). Placed at the end of
  Section 3.3, right after the paragraph describing pseudo-mask generation for the 100 unlabeled
  images, with a caption describing the MiT-B0 encoder's four feature scales (1/4, 1/8, 1/16,
  1/32), the All-MLP decoder's project/resize/fuse steps, the 5-class classifier at 1/4
  resolution, and the resize+argmax step producing the pseudo-mask - verified accurate against
  `src/segmentation/train.py`'s actual model (`SegformerForSemanticSegmentation.from_pretrained
  ("nvidia/mit-b0", num_labels=5, ...)`) and citation [14] (Xie et al. 2021, already used
  elsewhere in Related Work). Source: `fig2_segformer_architecture.{png,svg,pdf}`,
  `src/evaluation/plot_segformer_architecture.py`.
- [x] **Renamed every paper figure PNG to `fig<N>_<content>.png`, matching its actual Figure N
  in the current text - in two passes, because of a concurrent-edit collision.** First pass (after
  the Figure 3/gaussian-architecture insertion below shifted every later figure by one): renamed
  8 files via `git mv` to `fig1_pipeline` through `fig9_task_b_training`. While that pass's
  generator-script updates were still in progress, the concurrent Codex session above inserted
  the new Figure 2 (SegFormer) directly into `DRAFT.md` and reorganized `paper/figures/` on its
  own - moving `bridge_splat_real_color_{rgb,semantic}.png`, the gaussian-architecture figure's
  `.svg`/`.pdf` siblings, and `segformer-references.png` (unrelated, provenance unclear) into a
  new `paper/figures/tmp/` subdirectory, and deleting the just-renamed orphaned ablation figure
  (`unused_ablation_convergence_seed43.png`, itself a rename of the never-referenced
  `fig9_ablation_convergence.png` - see the "Removed the no-warmstart ablation" entry below) with
  no copy left anywhere findable. Caught this via `git status` showing unexplained deletions right
  before committing; paused and confirmed with the user rather than committing over a concurrent
  change. Second pass, after confirming the new Figure 2 placement was correct and intentional:
  renamed every figure again to its final number - `fig1_pipeline`, `fig2_segformer_architecture`,
  `fig3_cable_voting`, `fig4_gaussian_architecture` (only the `.png` survived the first pass -
  its `.svg`/`.pdf` siblings were in `tmp/` under the old `fig3_` name - so regenerated all three
  under the final name by re-running `plot_semantic_tensor_architecture.py`, byte-identical to
  the `tmp/` copies; the stale `tmp/` copies were left as-is, not deleted),
  `fig5_per_class_iou`, `fig6_qualitative_grid`, `fig7_interpolation`, `fig8_splat_render`,
  `fig9_task_a_training`, `fig10_task_b_training`. Updated all 10
  `![Figure N: ...](figures/...)` references in `DRAFT.md` and every generator script's default
  output filename (`plot_pipeline_diagram.py`, `plot_segformer_architecture.py`,
  `plot_cable_voting_figure.py`, `plot_semantic_tensor_architecture.py`, `plot_per_class_iou.py`,
  `plot_qualitative_grid.py`, `plot_interpolation_sequence.py`, `compose_splat_screenshots.py`,
  `plot_training_curves.py`). Net effect versus before this pair of passes: the orphaned ablation
  figure is now genuinely gone (not just moved) - it was never referenced by the paper and is
  reproducible via `plot_ablation_convergence.py` from the preserved
  `gaussians_replicate_{warmstart,no_warmstart}_seed43` checkpoints if ever needed again; nothing
  else was lost. `bridge_splat_real_color_{rgb,semantic}.png` (Figure 8's source screenshots, not
  embedded directly) and `segformer-references.png` now live under `paper/figures/tmp/`, untouched
  by either pass.
- [x] **Added the semantic Gaussian architecture figure (new Figure 3, Section 3.4)**, revisiting
  the previously-paused idea (two earlier attempts - a matplotlib shared-trunk/dual-head diagram
  and a Mermaid version - were reverted at the user's request; see the old Outstanding entry this
  replaces). The user asked for a written architecture description to hand to an external tool
  (Codex) rather than have me draw it directly; that description (per-Gaussian parameter table,
  init sources, fused-tensor concatenation, single rasterization pass, output split) became the
  spec Codex drew from. Codex produced two iterations: a simpler shared-trunk/dual-head diagram
  (`semantic_gaussian_architecture.*`, `src/evaluation/plot_semantic_architecture.py`), then a
  more detailed tensor-style version with real per-Gaussian attribute bars, the training-loss
  formula, and paired real RGB/semantic-map thumbnails
  (`semantic_gaussian_tensor_architecture.*`, `src/evaluation/plot_semantic_tensor_architecture.py`).
  Verified the tensor version's accuracy directly against `src/gaussian_splatting/model.py` (tensor
  shapes $N\times3$/$N\times5$/$N\times8$, sigmoid on RGB and opacity, raw semantic logits,
  `gsplat.rasterization` call) and `losses.py`/`train.py` (loss formula, mask-source weight $w$) -
  matches exactly. The RGB/semantic thumbnails are real renders of held-out test view 300
  (`outputs/renders/fig4_holdout_real_color/300_{rgb,sem}.png`, the same view discussed in
  Figure 5's qualitative results), not fabricated illustrations. User picked the tensor version;
  deleted the superseded simpler version's PNG/SVG/PDF and generator script. Inserted as new
  Figure 3 at the end of Section 3.4 (after the Representation/Semantic warm-start/Fused
  rendering/Losses paragraphs it visually summarizes, before Densification), renumbering old
  Figures 3-8 to 4-9 throughout the document (same ordered-sed technique used for the Table
  renumbering above).
- [x] **Refreshed Figure 6 (splat viewer render, Section 5.3, optional) for the fix #4 (color
  init) retrain.** The PLYs and composed figure on disk were all from before the fix #4 retrain
  (dated 04-09 Sep, checkpoint completed 10 Sep 17:49-17:53) - re-exported both
  `export_splat_ply`/`export_semantic_splat_ply` via `src/gaussian_splatting/export_ply.py` from
  the current canonical checkpoint (`outputs/checkpoints/gaussians/final.pt`,
  `bridge_splat_real_color_{rgb,semantic}.ply`), user captured true-color and semantic
  screenshots in SuperSplat, composed via `src/evaluation/compose_splat_screenshots.py` into
  `fig6_splat_render.png`. Deleted 9 stale `.ply` exports from `outputs/renders/` (~270MB,
  gitignored local artifacts, all reproducible from preserved checkpoints if ever needed again)
  and replaced the old `bridge_splat_{rgb,semantic}_v2.png` source screenshots in git with the
  new `bridge_splat_real_color_{rgb,semantic}.png` pair.
- [x] **Added a dedicated Task A (SegFormer) results table**, at the user's request after a
  back-and-forth about which "mIoU" Table 2 (formerly Table 1) actually reports. Section 5.1 was
  previously silent on Task A's own validation performance except for one number folded into
  Section 5.4's training-convergence prose (81.67%, next to Figure 7) - easy to mistake for, or
  conflate with, Table 2's structural mIoU (91.04%), despite the two being different tasks (2D
  segmentation vs. 3D render), different splits (30-image internal-val vs. 30-image test holdout),
  and different roles (checkpoint-selection diagnostic vs. the actual scored deliverable). Added a
  new **Table 1: Task A (SegFormer) per-class validation IoU** (deck 92.80%, stay_cable 89.17%,
  tower 78.20%, foundation 66.48%, background 98.64%, structural mIoU 81.67% - read directly from
  `outputs/checkpoints/segformer_mitb0/best.pt`'s stored `val_ious`/`val_miou`, no retraining
  needed since `train.py` already saves per-class IoU at checkpoint time) with an explicit
  disambiguating intro paragraph in Section 5.1, before the existing Task B tables. Renumbered the
  three existing tables (Overall holdout performance, Per-class IoU, Resolution ablation) from
  1/2/3 to 2/3/4 throughout the document (8 cross-references updated via a single ordered
  `sed` pass, 3->4 then 2->3 then 1->2, to avoid collisions) and added a cross-reference from
  Section 5.4's Figure 7 prose back to the new Table 1. Deliberately kept this as a separate table
  rather than added as rows/columns to Table 2/3 - Task A's number is not part of the scored
  deliverable and merging them risked implying otherwise.
- [x] **External review fix #4 (Gaussian color init) implemented and retrained.** `DRAFT.md`
  Section 3.4 claimed each Gaussian's initial RGB came from the triangulated point's own observed
  color, but `SemanticProjector.project()` (`src/colmap_io/semantic_voting.py`) actually assigned
  the voted-class's fixed 5-color legend swatch (`CLASS_COLORS`) as `point_colors`, and
  `PycolmapReconstructor`'s `Point3D` never stored real pixel color at all - so the paper's
  description and the code disagreed, and the initial point cloud was effectively colored by
  semantic class rather than by appearance. Fixed by adding `sample_point_colors()`
  (`src/colmap_io/reconstructor.py`) - for each 3D point, samples the real RGB pixel value from
  every observing *original* (pre-undistortion) image at that point's own projected 2D feature
  location (`ImagePose.points2d`, the same original-image coordinates triangulation and the
  semantic vote already use), averaged across observations, gray fallback for points with no
  readable observation. Deliberately not restricted to any train/val/test split - color is
  photometric input data, not a supervised label, the same treatment position already gets from
  triangulation (which uses all 400 images regardless of split). `src/gaussian_splatting/train.py`
  now calls `sample_point_colors` instead of reusing the semantic projector's palette colors.
  Added `tests/test_sample_point_colors.py` (5 new tests, synthetic solid-color images).
  - **Also corrected Section 3.4's "unlike standard practice" framing**: sparse-point-cloud color
    init *is* standard 3DGS practice (Kerbl et al. [6]); only the semantic-logit warm-start is
    this paper's own addition. Rewritten to state plainly that position and color both follow
    standard practice, and describe the (now-fixed) real-pixel-averaging color source.
  - **Retrained Task B once** (`outputs/checkpoints/gaussians_real_color_init/`, installed as
    canonical; prior palette-color checkpoint preserved as
    `gaussians_stale_palette_color_init_pre_20260910`) - 604,152 Gaussians (vs. 600,404 before,
    same 40,000-iteration budget and 80/10/10 split, only the color source changed), Gaussian
    count stabilizing at step 7,400 (vs. 8,200 before).
  - **Result on the same 30-image test set**: PSNR 22.13 dB (was 21.83), SSIM 0.853 (was 0.843),
    LPIPS 0.321 (was 0.349), mIoU 91.04% (was 88.97%; deck 92.59%, stay_cable 91.00%,
    tower 91.88%, foundation 88.71%, background 98.93%), Accuracy Score 0.816 (was 0.798). Every
    metric improved - a real photometric-color starting point converges to a better fit within
    the fixed 40k-iteration budget than the previous class-swatch-colored start, which the
    photometric loss (L1 + D-SSIM) had to spend part of the training budget correcting away from
    before it could refine geometry/appearance further; this is a plausible mechanism, not a
    formally isolated ablation. Per-class ranking changed: tower now edges out stay_cable
    (previously deck > cable > tower > foundation; now deck > tower > cable > foundation) - Table
    2 and Section 5.2's discussion were rewritten to describe this specific ranking rather than
    reusing the prior round's narrative template.
  - **Updated throughout `DRAFT.md`**: Abstract, Introduction (bullet 5), Section 3.4's
    warm-start paragraph, Gaussian count in Sections 3.4/4.2, Table 1, Table 2, Figure 3's
    surrounding text/caption, Section 5.2 (deck/cable/tower discussion), Section 5.4 (Gaussian-
    count-stabilization step 8,200 -> 7,400), Conclusion (mIoU/Accuracy Score). Regenerated
    Figures 1, 3, 4, 5, 8. Updated `README.md`'s results table and Gaussian count; while there,
    also corrected two numbers left stale by the earlier #1 (split) fix - Task B's own training-
    view count ("240 GT-mask + 100 pseudo-mask = 340 views" -> "270 + 100 = 370", since Task B
    trains on train+val, not train alone) and the evaluation holdout size ("60 held-out views" ->
    "30", in two places) - neither was part of review finding #4, just noticed while editing the
    same file.
  - Ran the full test suite after all changes.
- [x] **External review text fixes #3, #6, #7, #8, #9, #10, #11, and reference polish - all
  text-only, no retraining needed** (see the "External review" Done entry below for the original
  findings list). In the order fixed:
  - **#3 Bridge type**: title, abstract, Introduction, Section 3.1/3.4/4.1 all changed
    "cable-stayed bridge" -> "suspension bridge"; every plain-prose "stay cable(s)" (physical
    structure) -> "main cable and hangers"; kept `stay_cable` (backtick) only where referring to
    the dataset's own class name/annotation, with one explicit note (Introduction) that this is
    the dataset's JSON label while the contest brief itself calls it "main cable" (p.9). Section
    5.2's "fan of stay cables spans a... planar surface" argument rewritten to the geometrically
    correct suspension-bridge claim: the main cable and its hangers lie within a single
    near-vertical plane running the length of the span (verified this is still a valid,
    accurate description of suspension-bridge cable-plane geometry - the underlying "3D-consistent
    annotation" argument survives, just with the right structure name). Figure 2/6 captions
    updated too ("cable fan" -> "main cable"). Citations describing *other* papers' cable-stayed
    bridges (Hu et al. [2]'s actual subject) left unchanged - correctly describes that cited work,
    not our own bridge.
  - **#6 Strided-split rationale**: Section 3.6 rewritten to state plainly that the strided split
    does *not* avoid near-duplicate train/test adjacency (it guarantees a training frame next to
    almost every held-out frame) - its real, defensible benefit is spreading the holdout uniformly
    across the whole flight envelope rather than one segment. Added an explicit statement that the
    reported metrics measure interpolation within a dense trajectory, not extrapolation, and that
    a contiguous-block/leave-block-out split would be a stronger secondary evaluation (left as
    future work, not implemented).
  - **#7 Per-class explanations**: re-measured the reviewer's pixel-ratio and observation-count
    numbers against the *current* 30-image test split / 270-image training pool (they were
    originally measured against the old 60-image split) - the critique holds with fresh numbers:
    `stay_cable` is still the largest structural class by pixel count on the new test split
    (7.16% vs. `deck`'s 5.02%), and `deck` still has the lowest average observations per Gaussian
    (2.68 vs. `foundation`'s 4.66) despite scoring highest. Section 5.2 rewritten: cable's
    "thin, sparsely-sampled" framing now explicitly distinguishes "physically thin strands" from
    "the coarse, actually-largest-by-pixel-count annotated region" (both stated with numbers);
    foundation's explanation changed from the unsupported "fewer observing viewpoints" (directly
    contradicted by deck's own lower count) to a hedged "smallest absolute footprint (0.62% of
    test pixels) means IoU is more sensitive to fixed-width boundary error" hypothesis, explicitly
    flagged as unconfirmed (no dedicated ablation run).
  - **#8 Table 3 footnote**: reworded to say both rows *did* use the semantic warm-start (verified
    true - both logs show a real voted class distribution), differing only in voting rule
    (old strict-cable-majority vs. current plain-plurality) - and now also flags the footnote's
    other two known discrepancies from Table 1 (pre-mask-fix, pre-split-fix), consistent with the
    still-open "retrain Table 3" item in Outstanding.
  - **#9 Metric formulas**: PSNR's MSE now explicitly sums over the 3 color channels
    (`\sum_{c=1}^{3}`, `1/3HW` normalizer) instead of silently treating each pixel as scalar; SSIM
    range changed from a flat "`[0,1]`" to "in `[-1,1]` in general, close to `[0,1]` in practice
    here"; mIoU's formula text now states explicitly that `TP`/`FP`/`FN` are pooled from one
    confusion matrix over all 30 test views at once (pixel-weighted), not averaged per-image
    (Cityscapes-style, now named as such); LPIPS's "blurred cable strands" example now explicitly
    flagged as our own illustrative extrapolation, not a finding from Zhang et al. [18].
  - **#10 Convergence description**: Figure 8's "raw per-step loss"/"15-step moving average" ->
    "loss logged every 100 steps"/"15-sample... roughly 1,500 training steps"; Gaussian-count
    stabilization step and plateau-onset step re-verified against the new 80/10/10-split log
    (8,200 and ~20,000 respectively, both re-measured, not just carried over); Task A's "clean,
    monotonic curves" -> "smoother, less noisy curve" (re-checked on the new log: still 20/79
    epoch-to-epoch decreases, not literally monotonic); densification "capped at 600,000" ->
    explicit "soft cap... a single step can overshoot it" with the real final count as the example.
  - **#11 Rhetoric**: "digital twin" (Abstract, Conclusion) -> "semantic 3D scene representation" /
    "digital-twin-ready asset," with an explicit note in the Conclusion that the model lacks a
    true digital twin's real-time sensor linkage and physical simulation. "Genuinely arbitrary
    viewpoint" (Figure 5's intro) reworded to state plainly that the interpolated poses are still
    *within* the flown envelope (interpolation, not extrapolation) and that we have no ground
    truth to verify accuracy there. Related Work's "our model answers... for any viewpoint an
    inspector specifies" similarly qualified with where accuracy is/isn't actually verified.
  - **Reference polish**: ref [3]'s title corrected to the PDF's verbatim wording ("unmanned
    aerial vehicles light detection and ranging data imagery," not "unmanned aerial vehicle LiDAR
    and imagery"); ref [13] given its missing volume/pages (`40, 801–816`, same verification
    already done for ref [3] earlier); Semantic-NeRF [7]'s "was the first" -> "an early,
    influential approach" (the cited paper doesn't itself claim priority).
  - Ran the full test suite after each block of edits (still 92 passed throughout, since these
    are prose-only changes with no code touched).
- [x] **External review fix #1 (holdout leakage) implemented and retrained.** SegFormer's
  checkpoint selection (`src/segmentation/train.py`) was validating on the same 60 images later
  reserved as the final render-based evaluation holdout - a model-selection step must not touch
  its own eventual test set. Fixed by switching from a 240/60 (train/test) split to a proper
  three-way 240/30/30 (train/internal-val/test) split, at the user's explicit choice of an 80/10/10
  ratio (discussed and confirmed in conversation, weighing "matches existing 20% test-set
  convention" against "less data for Task A" before landing on 10%/10%): added
  `src.evaluation.metrics.train_val_test_split` (test held out first via the existing
  `trajectory_interleaved_split`, then val held out from the remainder the same way - verified by
  test to produce exactly the same `test_ids` a single direct `trajectory_interleaved_split` call
  would) as the new single source of truth, alongside the original two-way split (kept for
  anything that still legitimately wants it, e.g. `vote_consistency.py`'s standalone analysis).
  `src/segmentation/train.py` now validates/selects its checkpoint on the 30-image internal-val
  split only, never touching the 30-image test split; `src/gaussian_splatting/train.py`'s
  `prepare_training_data` (shared by Task B training and `render_metrics.py`) folds train+val
  (270 images) into Task B's own training pool, since Task B has no checkpoint-selection step to
  protect - only `holdout_ids` (test, 30) stays untouched, identical to what Task A's own training
  excludes from validation. All CLI entry points that used to take `--holdout-ratio` now take
  `--val-ratio`/`--test-ratio` (both default 0.10): `segmentation/train.py`,
  `gaussian_splatting/train.py`, `render_metrics.py`, `plot_per_class_iou.py`,
  `plot_ablation_convergence.py`, `vote_consistency.py`. Added `TestTrainValTestSplit` (4 new
  tests) to `tests/test_split_utils.py`.
  - **Retrained the full pipeline once**: Task A (`outputs/checkpoints/segformer_mitb0_80_10_10/`,
    best val mIoU 81.67% at epoch 69, vs. the old leaked 81.27%), pseudo-labels regenerated from
    it (`outputs/pseudo_masks_80_10_10/`), then Task B (`outputs/checkpoints/gaussians_80_10_10/`,
    600,404 Gaussians) - installed as canonical (old checkpoints preserved as
    `*_stale_leaky_split_pre_20260910`).
  - **Result on the new 30-image test set** (not directly comparable to the old 60-image numbers -
    different, smaller test set, not just a different training procedure): PSNR 21.83 dB, SSIM
    0.843, LPIPS 0.349, mIoU 88.97% (deck 92.31%, stay_cable 89.98%, tower 87.69%,
    foundation 85.92%, background 98.75%), Accuracy Score 0.798. Per-class ranking reverted to
    deck > cable > tower > foundation (matching the very first seed=42 numbers from earlier in
    this session), unlike the intermediate mask-fix-only result where cable briefly edged out deck.
  - **Updated throughout `DRAFT.md`**: Abstract, Introduction, Sections 3.2 (added the
    mask-undistortion description that was missing even after the #2/mask fix below), 3.3 (full
    rewrite - internal-val rationale), 3.6 (full rewrite - three-way split), 4.1, 4.2, 5.1, 5.2
    (numbers + reverted ranking), 5.3 (view 005 replaced with 010, since 005 moved into the new
    train split - and the descriptive text rewritten to match what's actually in the new render,
    since 010 turned out to have more RGB blur than 005 did, not less), 5.4 (Task A best-epoch
    note), Conclusion. Regenerated Figures 1, 3 (also fixed two real bugs in
    `plot_per_class_iou.py` found while regenerating: a hardcoded "60-View" title, and a bar-label/
    mIoU-dashed-line text collision when a class's IoU happens to land within ~1.5 points of the
    mIoU line - both now parameterized/fixed generally, not just patched for this run's numbers),
    4, 5, 7, 8. `README.md`'s pipeline description and results table updated too.
  - This fix does not change the organizers' actual blind-test score (their test images are
    disjoint from all of ours regardless of how we split internally) - discussed explicitly with
    the user. Its value is making the paper's *self-reported* numbers an honest, leak-free
    estimate of generalization, matching what the methodology section actually claims, rather
    than a claim the organizers' own reproducibility check (brief: "the organizing committee will
    further verify the reproducibility") could contradict.
- [x] **External review triaged and fix #1 (of the review's numbering, "#2" - the mask/image
  coordinate misalignment) implemented and retrained.** An external methodology review of
  `DRAFT.md` (11 numbered issues + reference checks) was independently re-verified point by
  point against the actual code/data before acting on anything - nearly every specific number
  the review cited was reproduced exactly (10.22% unobserved points, 85.53/5.01/7.19/1.73/0.54%
  per-class pixel ratios, 2.42/3.90/5.47/4.27 avg observations/Gaussian, 229/240 training images
  with foundation, 5.88px max undistort displacement, 0.9235 foundation raw-vs-undistorted IoU,
  the seed42/43 warm-start ablation reversal, DOI/volume/pages for ref [13]) - and two errors
  were found *in* the review itself (both in its reference-check section: it cited nonexistent
  filenames for the Zhang et al. [1] and Lin et al. [13] PDFs, though ref [13]'s DOI/volume/page
  data was still correct once matched to the real file). The full triaged list, including which
  points were confirmed, partially agreed with, or disputed, is preserved in this session's
  transcript; the still-open items are logged under Outstanding above.
  - **What was fixed**: `undistort.py`'s image-undistortion pass wasn't matched by an equivalent
    mask-undistortion pass - GT masks (`json_to_mask.py`, rasterized from polygons drawn on the
    *original* distorted photos) and Task A's pseudo-masks (`infer.py`, predicted on those same
    original photos) were both being paired, via `mask_lookup` in
    `src/gaussian_splatting/train.py::prepare_training_data`, with the *undistorted* images and
    the pinhole camera `K` - a systematic misalignment (~6px at this camera's k1, worst at frame
    edges) affecting the Task B semantic loss for all 340 supervised views and the holdout
    evaluation itself. Fixed by adding `undistort_mask`/nearest-neighbor remap to `undistort.py`
    and lazily caching undistorted copies (`outputs/undistorted_gt_masks/`,
    `outputs/undistorted_pseudo_masks/`) for `mask_lookup` to point at instead - `render_metrics.py`
    reuses the same `prepare_training_data` function, so this one fix corrects both training
    supervision and evaluation ground truth. The semantic warm-start's own vote step
    deliberately keeps using the *original* masks, since a 3D point's 2D observation coordinates
    come from COLMAP's own feature tracks (detected on those same original photos) - undistorting
    that mask would have introduced a *new* misalignment there.
  - **Retrained Task B once** with the fix (`outputs/checkpoints/gaussians_mask_undistort_fix/`,
    2804.7s, 603,757 Gaussians), installed as the new canonical `outputs/checkpoints/gaussians/`
    (old checkpoint preserved as `gaussians_stale_mask_misaligned_pre_20260910/`). Task A
    (SegFormer) did not need retraining - it already trains/infers entirely on original,
    consistently-distorted images+masks (undistortion only ever happens at the Task B stage),
    so it had no alignment bug of its own.
  - **Result**: mIoU 91.28% -> **89.81%** (real drop, not noise - both training supervision and
    eval ground truth are now correctly aligned with the camera geometry, at the cost of losing
    whatever the model had been fitting to a self-consistent-but-wrong label position for).
    PSNR/SSIM/LPIPS barely moved (22.19->21.86 / 0.849->0.846 / 0.335->0.340), as expected since
    the photometric loss and RGB/camera alignment were never affected by this bug. Per-class:
    `deck` dropped the most (95.72%->91.19%, -4.53) - plausibly because it's a long, thin band
    at a grazing angle, most sensitive to a few pixels of boundary shift - while `stay_cable`
    barely moved (92.36%->91.33%) and is now the *highest*-scoring structural class, ahead of
    `deck`. `tower`/`foundation` moved less (-0.18/-0.17).
  - **Updated throughout `DRAFT.md`**: Abstract, Introduction contribution #5, Table 1/2,
    Figure 3's caption/discussion, Section 5.2's opening (now leads with cable > deck, not
    deck > cable), Section 3.4's Gaussian count, Conclusion. Regenerated Figures 1 (Gaussian
    count), 3 (per-class IoU), 4 (qualitative grid - re-rendered 005/050/250/300 from the new
    checkpoint via `src.gaussian_splatting.render`, and switched its GT-mask comparison panel to
    `outputs/undistorted_gt_masks` to match), 5 (interpolation - re-rendered via
    `src.gaussian_splatting.interpolate`), 8 (Task B training curve, from the new log). Also
    updated `README.md`'s results table and pipeline description. Figure 7 (splat viewer)
    intentionally left as-is - it needs a new manual SuperSplat capture, not yet done.
  - Section 5.2's specific explanatory claims (the "thin, sparsely-sampled" framing for cable,
    the "fewer viewpoints" framing for foundation) were deliberately **not** rewritten as part of
    this fix, even though the review's #7 already flags them as data-contradicted - that's
    scoped as a separate, later fix so the numbers-only update here stays reviewable on its own.
- [x] Full numeric audit of `DRAFT.md` (every reported number re-traced to a real checkpoint,
  eval report, log, or freshly-reproduced script run) - caught and fixed one real methodological
  bug: Table 3's (resolution ablation, Section 5.5) "Half" row was trained for only 30,000
  iterations (`gaussians_v1_halfres_noposeopt`, no training log ever saved for it - only
  recovered via the checkpoint's own `step` field and file timestamps) while the "Full" row used
  40,000 (`gaussians_v3a_fullres_noposeopt`) - not the "comparable iteration budget" the old text
  claimed, confounding the resolution-only conclusion. Retrained half-resolution for the full
  40,000 iterations (`--strict-cable-majority`, matching v3a's setup exactly so resolution is the
  only remaining difference) -> `outputs/checkpoints/gaussians_halfres_40k/final.pt` (601,144
  Gaussians, 1115.5s train time), evaluated via `render_metrics.py` ->
  `outputs/eval/render_eval_report_halfres_40k.md`. Result: half-resolution at 40k iterations
  scores *lower* (85.77% mIoU) than the old, shorter 30k-iteration run (87.96%) - an unexpected
  but real finding, disclosed in Table 3's discussion rather than hidden. This widens the
  measured resolution effect from +3.5 to +5.7 mIoU points and makes the "≈47 min vs ≈X min"
  training-time comparison literally apples-to-apples for the first time (≈47 min vs ≈19 min,
  both 40k iterations). Also confirmed via direct re-run/re-derivation that every other number in
  the paper is accurate: Table 1/2 (`render_eval_report_plain_plurality.md`), 600,958 Gaussians,
  86,336 raw feature tracks / 84,613 points after IQR filtering / 0.50px mean reprojection error
  (re-ran `python -m src.colmap_io.reconstructor` fresh), camera intrinsics (f=925.70, k1=0.00899,
  1320x989), 400 images (300 labeled/100 unlabeled), every Task A/B hyperparameter in Section 4.2
  against `train.py`/`src/segmentation/train.py` code, Task A's 81.27% final validation mIoU
  (`segmentation_train.log`), and the Gaussian count stabilizing at step 8,600
  (`gaussian_train_ablation_plain_plurality.log`).
- [x] Figure 6 (splat viewer, Section 5.3, optional) re-captured from the official
  plain-plurality checkpoint's PLYs in SuperSplat and recomposed via
  `src/evaluation/compose_splat_screenshots.py` (`paper/figures/fig6_splat_render.png`), replacing
  the stale strict-majority-checkpoint screenshots. Source screenshots
  (`paper/figures/bridge_splat_{rgb,semantic}_v2.png`) kept alongside the composed figure.
- [x] Removed the no-warmstart ablation (Section 5.2, old Table 3, old Figure 4/9) from the
  paper entirely — reverted to reporting the single seed=42 run (`gaussians_ablation_plain_plurality`
  checkpoint, 600,958 Gaussians) throughout, with the semantic warm-start now presented as a
  plain design choice (Section 3.4) with no accompanying ablation claim. Between this decision
  and the previous "removed the strict-majority ablation" entry above, two further attempts were
  made and abandoned:
  1. Ran N=2 same-config replicate trainings (`--seed` CLI flag added to
     `src/gaussian_splatting/train.py`, default 42) to measure the real run-to-run noise floor
     for warm-start vs. no-warmstart at a second seed (43). Result: 0.22 and 0.42 mIoU-point
     spreads between same-config seeds, both larger than the 0.17-point between-config
     difference the original ablation was built on — logged here only, never written into
     `DRAFT.md`. Checkpoints (`outputs/checkpoints/gaussians_replicate_{warmstart,no_warmstart}_seed43/`)
     and eval reports (`outputs/eval/render_eval_report_replicate_{warmstart,no_warmstart}_seed43.md`)
     kept on disk but unused.
  2. Attempted to fully replace the paper's reported numbers with the seed=43 replicate
     (requested explicitly), and discovered the mIoU-vs-training-step convergence *pattern*
     itself reverses between seeds — no-warmstart leads for most of training at seed=43, the
     opposite of seed=42's curve the original Figure 4/9 discussion was built on. This — combined
     with the noise-floor measurement in (1) exceeding the effect being discussed — was the
     deciding evidence for dropping the ablation narrative altogether rather than reporting
     either seed's ablation result as if it were stable.
  Concretely, this final change: reverted the Abstract/Table 1/Table 2/Figures 1, 3, 4, 5, 8
  (old numbering) to the seed=42 checkpoint's real numbers (mIoU 91.28%, deck 95.72%, cable
  92.36%, tower 89.71%, foundation 87.34%); deleted old Table 3, the old Figure 4 (mIoU
  convergence) embed/caption, and every "we tested with/without warm-start" paragraph in
  Section 5.2, Introduction, and the Conclusion; kept and reframed the still-valid,
  ablation-independent explanation for cable's high IoU (the coarse annotation is 3D-consistent,
  so the model reproduces the same annotation convention across viewpoints — not evidence of
  correcting toward truer geometry, verified against the qualitative render grid); renumbered
  Figures 5-9 -> 4-8 and Table 4 -> Table 3 throughout (single-pass placeholder-token regex, same
  technique as the earlier reading-order fix); fixed three leftover "majority vote"/"majority-voted"
  mentions in the Abstract, Introduction, and Related Work to "plurality vote"/"plurality-voted",
  matching Section 3.4's terminology (a stale leftover from the earlier plain-plurality baseline
  switch, unrelated to this pivot but caught while auditing the same text).
  Now-orphaned artifacts kept on disk but unused by the paper: `fig9_ablation_convergence.png`
  (the old Figure 4 image, seed=43 version — last thing rendered before this pivot),
  `outputs/checkpoints/gaussians_replicate_{warmstart,no_warmstart}_seed43/`,
  `outputs/renders/{fig4_holdout,fig5_interp}_seed43/`, `bridge_splat_seed43_{rgb,semantic}.ply`.
  `src/evaluation/plot_ablation_convergence.py` and `tests/test_plot_ablation_convergence.py` are
  kept as-is (not deleted) — same precedent as `vote_consistency.py`: a real, tested, standalone
  tool no longer cited by the paper rather than dead code.

- [x] **Measured the real noise floor** for Table 3's "within noise" claims, instead of asserting
  it from general experience: ran one same-config, different-seed replicate for each of Table
  3's two configs (`--seed 43` vs. the original `--seed 42`, both 40,000 iters/full-res,
  `src/gaussian_splatting/train.py`'s new `--seed` CLI flag). Results:
  - Warm-start: seed 42 -> 91.28% mIoU, seed 43 -> 91.06% mIoU (checkpoint
    `outputs/checkpoints/gaussians_replicate_warmstart_seed43/`, eval
    `outputs/eval/render_eval_report_replicate_warmstart_seed43.md`). Within-config spread:
    **0.22 points**.
  - No-warmstart: seed 42 -> 91.45% mIoU, seed 43 -> 91.03% mIoU (checkpoint
    `outputs/checkpoints/gaussians_replicate_no_warmstart_seed43/`, eval
    `outputs/eval/render_eval_report_replicate_no_warmstart_seed43.md`). Within-config spread:
    **0.42 points**.
  - Both same-config (different-seed) spreads (0.22, 0.42) are *larger* than the 0.17-point
    between-config difference Table 3 reports (91.45% vs. 91.28%) - real, measured support for
    the "within noise" language already used throughout Section 5.2 and Table 4's footnote,
    not just an assumption from general ML experience. Per-class deltas show the same pattern
    (e.g. `deck` alone varies 0.96 points across seeds in the no-warmstart config - larger than
    the 0.91-point `foundation` gap Table 3's discussion calls "the largest single-class gap
    in the table").
  - N=2 per config is not enough for a real std/CI (that would need 3+), but as a sanity check
    it's more than sufficient: it directly falsifies the concern that the between-config
    difference might be a real effect rather than noise.
  - **Not yet incorporated into `DRAFT.md`** - this was run specifically as a background
    verification per explicit request, not to add new rows to Table 3 or new curves to Figure 4.
    If the paper should cite this noise-floor measurement directly (e.g. a footnote on Table 3
    or in the "within noise" sentences), that's a follow-up ask, not done automatically.
- [x] **Fixed a real submission-blocking bug**: `outputs/checkpoints/gaussians/final.pt` - the
  canonical default path `train.py` writes to with no `--output-dir` flag, and what
  `README.md`'s documented Quickstart commands (`render.py`, `render_metrics.py`) point at - was
  still the OLD strict-majority checkpoint (602,363 Gaussians, step 40,000, dated Sep 4), not the
  current official plain-plurality one. Anyone (including the organizers) following the
  README's exact documented commands would have evaluated/submitted the wrong model. Fixed by
  renaming the stale folder to
  `outputs/checkpoints/gaussians_stale_strict_majority_pre_20250908/` (kept, not deleted) and
  copying `gaussians_ablation_plain_plurality/` into `gaussians/` so the canonical path now holds
  the real official checkpoint (600,958 Gaussians). Re-ran `render_metrics.py` against the new
  `gaussians/final.pt` and confirmed it reproduces Table 1/2 exactly (mIoU 91.28%, cable 92.36%,
  deck 95.72%, tower 89.71%, foundation 87.34%). This is purely an `outputs/` (gitignored)
  filesystem fix - no paper or source-code changes.

- [x] Changed Figure 4 (Section 5.2) from plotting `stay_cable`-only IoU to structural mIoU
  convergence, since the surrounding discussion had broadened from cable specifically to the
  full per-class/mIoU picture (the earlier "explicitly state mIoU is lower with warm-start"
  entry below). `plot_ablation_convergence.py`'s `plot_ablation_convergence()` no longer takes a
  `class_id` param - it always computes mIoU via `compute_miou` from each checkpoint's per-class
  IoU dict. The real re-rendered curve confirms the text: warm-start leads by 1.8 mIoU points at
  step 2,000, the two curves are within 0.1 points by step 24,000, then interleave within noise
  through 40,000 (no-warmstart ending marginally ahead) - matches Table 3 exactly.

- [x] Fixed figure numbering out of reading order: the training-convergence ablation figure had
  been added to Section 5.2 as "Figure 9" (since it was the 8th figure chronologically added to
  the paper), but Section 5.2 appears *before* Section 5.3's Figures 4-6 and Section 5.4's
  Figures 7-8 in reading order - so "Figure 9" was appearing on the page before "Figure 4". Ran
  a single-pass renumber across all in-text "Figure N" references (old 9 -> 4, old 4-8 -> 5-9)
  so numbers now strictly increase in reading order; Table numbers (1-4) were already correct.
  PNG filenames were left as-is (e.g. `fig9_ablation_convergence.png` is now displayed as
  "Figure 4") - only the in-paper label moved, which is normal and doesn't need to match the
  filename, but worth knowing if grepping the repo by figure number.
- [x] Removed a stray code-artifact mention in Section 5.2 body prose (a literal
  `` `outputs/gt_masks/` `` path reference) - same category of issue as the earlier
  export_splat_ply/skimage.metrics cleanup, caught on a fresh read-through.

- [x] Removed the strict-majority cable-voting rule from the paper entirely, per explicit
  request, rather than keeping it as a discussed-and-rejected alternative: dropped it from
  Table 3 (now 2 rows: our approach vs. no-semantic-warmstart only) and Figure 9 (now 2 curves),
  and from every other mention (Section 3.4, Introduction contributions, Conclusion, Table 4's
  footnote, the Lin et al. Related Work comparison). `src/evaluation/plot_ablation_convergence.py`
  simplified to match (removed the `strict_majority` curve/CLI arg).
  `src/evaluation/vote_consistency.py` (the script computing the 416-points/92.3%-to-background
  stats, no longer cited anywhere in the paper) is kept as a standalone diagnostic - its
  docstring no longer claims to feed the paper. The `strict_cable_majority=True` code path and
  `--strict-cable-majority` CLI flag are still functional (real, tested, harmless to keep) even
  though no paper text describes them anymore.
- [x] Switched the paper's official model from the strict-majority baseline to the
  plain-plurality one, since the (now-removed) strict-majority ablation showed that rule doesn't
  improve on plain plurality where it matters (cable IoU) despite adding cable-specific
  complexity. Regenerated Figures 1, 2, 3, 4, 5, 8, 9 and Tables 1/2/3 from the plain-plurality
  checkpoint (`outputs/checkpoints/gaussians_ablation_plain_plurality/final.pt`, 600,958
  Gaussians). Flipped the actual code default too
  (`vote_majority_class`/`SemanticProjector.project`/`train()`'s `strict_cable_majority` is now
  `False` by default; CLI flag renamed `--plain-plurality` -> `--strict-cable-majority`,
  opt-in to the alternative) - this surfaced and fixed a real bug in
  `src/evaluation/vote_consistency.py`, which called `vote_majority_class()` without
  `strict_cable_majority=True` and would have silently started comparing plain plurality
  against itself; re-ran it and confirmed it still reproduces the exact numbers already cited
  in the paper (416 reclassified, 92.3% to background). Honestly reports that the rejected
  strict-majority alternative shows a higher tower IoU and marginally higher overall mIoU than
  our simpler default (Table 3) - flagged as an open, unexplained pattern rather than hidden.
  Backup of the pre-change state tagged `pre-plurality-simplification` (commit `dc9f8de`).
- [x] Ablation training without the strict-majority rule (plain plurality only) and without
  the semantic warm-start entirely (`DRAFT.md` Section 5.2, Table 3) - two real 40,000-iteration,
  full-resolution runs (`--plain-plurality` and `--no-semantic-warmstart` flags added to
  `src/gaussian_splatting/train.py`), evaluated via `src/evaluation/render_metrics.py`
  (`outputs/eval/render_eval_report_plain_plurality.md`,
  `outputs/eval/render_eval_report_no_warmstart.md`). Result: neither mechanism has a
  measurable effect on cable's final IoU (92.36% and 92.02% vs. 92.13% baseline, within noise) -
  this contradicted the hypothesis in an earlier draft of Section 5.2, which has been rewritten
  to report the ablation honestly instead.
- [x] Figure 9 (cable IoU vs. training step for the ablation, Section 5.2, optional) - added
  after asking "is the null result in Table 3 a bug in the metric/methodology?": evaluated
  intermediate checkpoints (every 2,000-8,000 steps) of all three Table 3 runs via
  `src/evaluation/plot_ablation_convergence.py`. Confirms no bug - the warm-start/voting-rule
  mechanisms give a real, measurable early-training advantage (2.3 IoU points at step 2,000)
  that converges away by ~step 24,000, well inside the 40,000-iteration budget used for Table 3.

- [x] Fill in final Gaussian count in Section 3.4 (602,363 per
  `outputs/checkpoints/gaussians/final.pt`).
- [x] Figure 3 (per-class IoU bar chart, Section 5.1) - generated by
  `src/evaluation/plot_per_class_iou.py` from the real `render_eval_report.md` numbers.
- [x] Figure 4 (qualitative render grid, Section 5.3) - 4 real held-out views (005, 050, 250,
  300; verified against the official 60-view holdout list, not arbitrary picks) rendered via
  `src.gaussian_splatting.render` and assembled by `src/evaluation/plot_qualitative_grid.py`.
- [x] Figure 5 (novel-view interpolation filmstrip, Section 5.3) - SLERP/LERP path between real
  flown poses 280/300 via `src/gaussian_splatting/interpolate.py`, assembled by
  `src/evaluation/plot_interpolation_sequence.py`.
- [x] Figure 1 (pipeline overview, Section 3) - flowchart generated by
  `src/evaluation/plot_pipeline_diagram.py`.
- [x] Figure 2 (cable background-bleeding + voting schematic, Section 3.4) - real GT-mask overlay
  crop (image 300) + illustrative voting schematic, generated by
  `src/evaluation/plot_cable_voting_figure.py`.
- [x] Figure 6 (splat viewer render, Section 5.3, optional) - re-exported both
  `export_splat_ply` and `export_semantic_splat_ply` from the current official checkpoint (the
  old PLY on disk was stale, from an earlier run - 603,295 vs. the correct 602,363 points),
  manually captured true-color and semantic screenshots in SuperSplat
  (https://superspl.at/editor), composed side-by-side via
  `src/evaluation/compose_splat_screenshots.py`. (`plot_splat_pointcloud.py`'s static
  point-cloud-scatter render kept in the repo as a documented fallback for when no interactive
  viewer is available, no longer used for the paper figure itself.)
- [x] Figure 7 (Task A training convergence, Section 5.4) - loss + validation mIoU per epoch,
  parsed directly from `outputs/logs/segmentation_train.log` (the real 80-epoch run) by
  `src/evaluation/plot_training_curves.py`.
- [x] Figure 8 (Task B training convergence, Section 5.4) - per-step loss + moving average,
  parsed directly from `outputs/logs/gaussian_train_v3a.log` (the real run behind the official
  602,363-Gaussian checkpoint) by the same script. **All 8 figures now done.**
- [x] Restructure all 8 figures and 3 tables to IEEE convention: short caption on the figure/
  table itself, interpretation/analysis moved into body-text paragraphs with explicit
  "Figure N shows..." / "Table N shows..." cross-references.
- [x] Strengthen Related Work with real citations (Section 2 now cites 18 verified real papers,
  including SAM and LSeg added for the two foundation models named in the Feature 3DGS
  discussion; PDFs in `paper/references/`).
- [x] Verify Task A/B hyperparameters in Section 4.2 against the actual training code —
  caught and fixed one real error (Task A learning rate was stated as 6e-4, actually 6e-5).
- [x] Write the Abstract last, after Results is locked.
