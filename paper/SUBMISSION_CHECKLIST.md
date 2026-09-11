# Submission preparation

This checklist accompanies the working manuscript; it is not manuscript content.

## Required before submission

- [ ] Supply author names, affiliations, and the registered team identifier.
- [ ] Obtain the official IC-SHM manuscript template and format the paper within 10–15 pages.
- [ ] Inspect every page of the final PDF: equations, figure readability at print size, captions,
  table placement, reference formatting, and page count.
- [ ] Confirm the manuscript repository URL is publicly accessible and identify the submission
  commit. The URL currently names the project repository; access has not been reverified here.
- [ ] Package the reported checkpoint and record its checksum, environment, and rendering command.
- [ ] Verify the organizer's test-camera convention, semantic class mapping, and handling of
  background in mIoU against the final evaluation instructions. Local evaluation uses undistorted
  pinhole images and structural four-class mIoU.
- [ ] Run the documented reconstruction/rendering workflow in a clean environment with CUDA.
- [ ] Supply the shareable reproduction-data link required by the brief.
- [ ] Prepare the ten-minute English presentation video, with both slides and speaker visible,
  and its PowerPoint slides.

The competition brief in `data/Contest Dataset/` defines the paper, video/slides, code/README,
and reproduction-link requirements. This checklist does not assert that missing artifacts
have been created or that the organizer has approved an evaluation convention.

## Reproduce the revised manuscript assets

Run from the repository root in the project environment:

```bash
python -m src.evaluation.paper_statistics
python -m src.evaluation.plot_pipeline_diagram --horizontal --output paper/figures/fig1_pipeline.png
python -m src.evaluation.plot_cable_voting_figure --output paper/figures/fig3_cable_voting.png
python -m src.evaluation.plot_per_class_iou --report outputs/eval/render_eval_report_test_excluded_init.md --output paper/figures/fig5_per_class_iou.png
```

The statistics command writes `outputs/eval/paper_statistics.json`. Mask areas use undistorted
test masks; track counts distinguish all retained RGB observations from labeled semantic votes.
Figure 3 defaults to the aligned undistorted photograph and mask. Figure 5 reads the saved
evaluation report; this regeneration does not rerun model evaluation or training.
