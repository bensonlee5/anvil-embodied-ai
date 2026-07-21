# OpenARM2 33-episode stage-quality review v2

This directory contains the criterion-based blind rescore of all 99 stage
outcomes from the 33 successful T-shirt-fold demonstrations. The detailed
rubric and review protocol are in `docs/shirt-fold-labeling-v2.md`.

## Review sequence

1. The v1 manifest supplied only stage boundaries. Episode identity, v1 score,
   split, duration, retry count, and smoothing label were hidden.
2. Within each stage, anonymous tokens were shuffled and scored from four base
   frames near the stage outcome. Four stage-specific criteria were scored
   0–2; their sum maps mechanically to quality 1–5.
3. The 99 raw rows were validated before the token mapping was opened.
4. After unblinding, all five medium-visibility items and all 22 items more than
   one point from v1 received a five-second, three-camera temporal review.
5. Six items had criterion-level corrections. The other 21 temporal reviews
   confirmed the blind scores.

The pass is blind to the old numbers but is not independent inter-rater
evidence: the same AI-assisted reviewer performed the blind and temporal
reviews and had previously seen this dataset. Every final label therefore uses
`final_label_confidence=medium`, even when visibility was high.

## Files

- `blind_stage_quality_raw_v2.csv`: immutable 99-row blind response before
  episode identities or v1 labels were revealed.
- `blind_stage_mapping_v2.json`: deterministic seed-20260720 token mapping and
  reviewed frame coordinates.
- `episode_stage_quality_v2.csv`: joined old, raw, and adjudicated labels; raw
  and final criterion scores; review reasons; and retained/source intervals.
- `adjudication_report_v2.json`: distributions, agreement statistics,
  provenance hashes, and all six changes.
- `configs/training/priority_manifests/openarm2_shirt_fold_3stage_v2.json`:
  generated conservative sampler artifact.

The nine large anonymous image sheets are reproducible but intentionally not
checked into Git. Rebuild them in a fresh directory before a future blind pass:

```bash
uv run python scripts/training/build_openarm2_shirt_fold_blind_review.py \
  --output-dir=/tmp/openarm2-shirt-fold-blind-review
```

Regenerate or check the derived CSV, report, and manifest with:

```bash
uv run python scripts/training/build_openarm2_shirt_fold_priority_manifest_v2.py
uv run python scripts/training/build_openarm2_shirt_fold_priority_manifest_v2.py --check
```

The v1 annotations and manifest remain frozen historical artifacts.

## Result and training interpretation

Adjudicated mean quality is 3.58 for `side_one`, 3.52 for `side_two`, and 2.30
for `bottom_to_top`. The final stage contains 11 score-1 and 11 score-2
outcomes, despite all 33 episodes being task successes.

Agreement with the old holistic labels is weak for `side_one` (quadratic
weighted kappa 0.07) and `side_two` (0.28), but materially better for
`bottom_to_top` (0.68). This does not prove the new side labels are wrong: the
old labels had no criterion rubric. It does mean neither side-label set should
be treated as ground truth or used with aggressive sampling ratios.

The v2 sampler is intentionally conservative:

- score log-priorities are `[-0.4, -0.2, 0, 0.2, 0.4]`, a maximum 2.23x ratio;
- natural stage-frame mass is preserved on the exact seed-1000 27-episode
  training split instead of forcing three equal thirds;
- all retry candidates are neutral until they are semantically classified;
- smoothing is neutral; and
- the ordinary unweighted behavior-cloning loss is unchanged.

The manifest is suitable for a controlled data-selection experiment, not as a
production reward model. A genuinely production-quality label set still needs
an independent blinded rater and an explicit disagreement adjudication pass.
