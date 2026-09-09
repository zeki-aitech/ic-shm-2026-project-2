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
- [ ] Figure 6 (splat viewer, Section 5.3, optional) still shows the OLD strict-majority
  checkpoint's SuperSplat screenshots (`paper/figures/fig6_splat_render.png`). New PLYs from the
  official plain-plurality checkpoint are already exported
  (`outputs/renders/bridge_splat_plain_plurality_{rgb,semantic}.ply`, via the new
  `src/gaussian_splatting/export_ply.py`) - needs manual capture in SuperSplat
  (https://superspl.at/editor) and re-composing via
  `src/evaluation/compose_splat_screenshots.py`. Geometry is very close between the two
  checkpoints (Table 3), so this is a low-urgency accuracy nit, not a correctness bug.
- [ ] `docs/EXPERIMENT_PROGRESS_AND_FINDINGS.md` and `docs/SUBMISSION_CHECKLIST_AND_GUIDELINES.md`
  still reference the old strict-majority checkpoint's numbers/Gaussian count (602,363) - not
  updated in the plain-plurality baseline switch below (only `DRAFT.md` and `README.md` were).

## Done

- [x] Switched the paper's official model from the strict-majority baseline to the
  plain-plurality one, since Table 3's ablation showed the strict-majority rule doesn't improve
  on plain plurality where it matters (cable IoU) despite adding cable-specific complexity.
  Regenerated Figures 1, 2, 3, 4, 5, 8, 9 and Tables 1/2/3 from the plain-plurality checkpoint
  (`outputs/checkpoints/gaussians_ablation_plain_plurality/final.pt`, 600,958 Gaussians);
  Table 3's rows now read as "our approach" (plain plurality) vs. two tested alternatives
  (strict-majority rule, no semantic warm-start) instead of "baseline" vs. two ablations of
  itself. Flipped the actual code default too
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
