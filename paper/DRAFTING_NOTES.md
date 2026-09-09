# Drafting Notes — IC-SHM 2026 Project 2 Paper

`DRAFT.md` is a structural/content draft of the paper (target: 10-15 pages, official IC-SHM
template, English) — it is NOT yet formatted in the official template. Pour its content into
the downloaded template once available from the contest website, before final submission.

This file tracks outstanding work and a record of what's already been done. Delete or archive
it once the paper is finalized and submitted — it is not part of the paper itself.

## Outstanding

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

## Done

- [x] New Figure 3 (Section 3.4, "Fused rendering"): a schematic of the semantic Gaussian
  representation and fused single-pass rasterization - the paper's own central contribution
  (per-Gaussian semantic logits + fused RGB/semantic rasterization), unlike NeRF/3DGS/SegFormer
  which are third-party architectures described in text only (Related Work), not diagrammed.
  Illustrative, not tied to one real Gaussian/checkpoint, in the same spirit as Figure 2's
  right-side voting schematic. Built with
  `src/evaluation/plot_gaussian_representation_figure.py` ->
  `paper/figures/fig3_gaussian_representation.png`. Figures 3-8 renumbered to 4-9 throughout
  (same placeholder-token regex technique as earlier renumbering passes) - caught and fixed two
  places the regex missed because the old number wasn't a plain "Figure N" match: "Figure\n4's"
  (a mid-sentence line-wrap put a newline where the regex expected a space) and three plural
  "Figures N and M" / "Figures N-M" references, all now correct after a manual pass.
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
