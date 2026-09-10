# Drafting Notes — IC-SHM 2026 Project 2 Paper

`DRAFT.md` is a structural/content draft of the paper (target: 10-15 pages, official IC-SHM
template, English) — it is NOT yet formatted in the official template. Pour its content into
the downloaded template once available from the contest website, before final submission.

This file tracks outstanding work and a record of what's already been done. Delete or archive
it once the paper is finalized and submitted — it is not part of the paper itself.

## Outstanding

- [ ] External-review findings not yet acted on (see the "External review" Done entry below for
  the full list this was triaged from):
  - [ ] **#4 Gaussian color init doesn't match its own description**: `DRAFT.md` Section 3.4 says
    initial RGB comes from the triangulated point's own color, but `SemanticProjector.project()`
    (`src/colmap_io/semantic_voting.py:187`) actually assigns the voted-class palette color
    (`CLASS_COLORS`), and `PycolmapReconstructor`'s `Point3D` never stores real point color at
    all - fix the init to sample real color from an observing image, or fix the paper's
    description (and drop the "unlike standard practice" framing around sparse-point init
    itself, since that part **is** standard 3DGS practice per Kerbl et al. 2023 - only the
    semantic-logit warm-start is this paper's own addition). Requires retraining Task B.
  - [ ] Table 3 (resolution ablation) still uses the pre-mask-fix, pre-split-fix checkpoints
    (`gaussians_v3a_fullres_noposeopt`, `gaussians_halfres_40k`) - same mask/image coordinate
    misalignment fixed below, and the old 240/60 split rather than the current 240/30/30. Low
    priority (it's a secondary ablation, not the headline numbers) but should eventually be
    redone for full consistency.
- [ ] Confirm exact page/formatting requirements once the official template is downloaded from
  the IC-SHM website and available locally.
- [ ] Reformat citations into the official template's required style once available
  (`DRAFT.md` References section) — if the template mandates author-year, both the in-text
  citations and the reference list's ordering will need to change accordingly.
- [ ] Team/author details: name(s), affiliation, IC-SHM 2026 Project 2 team identifier
  (`DRAFT.md` title block, currently `[NEEDS]`).
- [ ] Repository URL for the Code Availability sentence (`DRAFT.md` Section 1, currently
  `[NEEDS]`) once the submission link is finalized.
- [ ] `docs/EXPERIMENT_PROGRESS_AND_FINDINGS.md` and `docs/SUBMISSION_CHECKLIST_AND_GUIDELINES.md`
  still reference the old strict-majority checkpoint's numbers/Gaussian count (602,363) - not
  updated in the plain-plurality baseline switch below (only `DRAFT.md` and `README.md` were).
- [ ] Revisit adding a Figure 3 for the semantic Gaussian representation / fused rasterization
  (Section 3.4) - tried a shared-trunk, dual-head matplotlib diagram (per-class-colored channel
  bars for the 5 semantic logits, geometric params bypassing the concatenation into the shared
  `gsplat` trunk) and a Mermaid version of the same content; reverted both (commits `f78e05b`,
  `3943a70`, `099c43e`) at the user's request to pause on this rather than reject the idea
  outright - the "this paper's addition" vs. standard-3DGS framing itself is still considered
  worth a figure, just not this execution.

## Done

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
