# OpenArm2 shirt-fold labeling v2

This document freezes the second-pass labeling protocol for the 33 successful
OpenArm2 T-shirt-fold demonstrations. It separates three questions that must not
be collapsed into one score:

1. **What task stage is being executed?**
2. **What visible garment state did that stage produce?**
3. **What manipulation behavior produced it?**

The policy-training dataset remains the 34,850-frame phase-aligned trim. The
43,625-frame source videos may be used to inspect additional temporal context,
but labels are reported in both retained and source frame coordinates.

## Blind review protocol

The second pass is blind to the v1 quality scores:

- Review items use anonymous tokens and are shuffled independently within each
  stage.
- The reviewer knows the stage because each stage has a different rubric, but
  does not see the episode index, split assignment, prior score, duration,
  retry count, or smoothing label.
- Each item initially shows four base-camera frames from the final 1.5 seconds
  of the stage. Wrist views and a short full-rate temporal window are consulted
  only when the base view is ambiguous.
- All four criterion scores and visibility confidence are recorded before the
  anonymous token is resolved back to an episode.
- After the initial pass, items are compared only within the same stage and
  score band. A score changes only when a criterion score is corrected; the
  distribution is never forced into quotas.

This is a blinded rescore by the same AI-assisted reviewer that previously saw
the dataset, not an independent second human annotator. It reduces anchoring to
the old numeric labels but does not establish inter-rater reliability.

## Rules shared by all stages

Quality describes the visible garment state produced by the stage, not how the
robot moved:

- All 33 demonstrations are task successes. Score 1 means a poor-quality
  successful intermediate or final fold, not task failure.
- Slow, deliberate smoothing is quality-neutral. It receives credit when it
  leaves a visibly flatter or better-aligned result, but slow motion itself is
  neither rewarded nor penalized.
- A repeated close/reopen/retry is not automatically a misgrasp. Retry count,
  path length, elapsed time, and arm speed are excluded from the quality score.
- Robot pose and final shirt position on the table are ignored unless they
  prevent the garment state from being observed.
- Wrinkles inherent to the starting shirt are ignored when the stage does not
  worsen them. New bunching, twists, displaced layers, and protrusions count.
- The state immediately before the next topology-changing grasp is the stage
  outcome. Approach motion for the next stage must not be judged as part of the
  previous fold.

### Criterion scale

Every stage has four criteria, each scored 0, 1, or 2:

- **2 — strong:** the intended geometry is clear, stable, and has at most a
  small localized defect.
- **1 — acceptable:** the stage is successful but has a visible, material
  defect that does not dominate the result.
- **0 — weak:** the criterion is substantially missed, unstable, or dominated
  by bunching, misalignment, or an uncontained layer.

The four-criterion total maps mechanically to the ordinal quality score:

| Criterion total | Quality score | Interpretation |
|---:|---:|---|
| 0–1 | 1 | Poor-quality success; intended state is barely stable |
| 2–3 | 2 | Clearly completed but substantially defective |
| 4–5 | 3 | Usable intermediate/final state with moderate defects |
| 6–7 | 4 | Good result with only localized defects |
| 8 | 5 | Strong on every criterion |

An incomplete or wrong-topology fold is score 1 regardless of the total. A
whole sleeve or side remaining outside the intended fold caps the result at 2.
These guards do not turn a successful episode into a failure label.

### Visibility confidence

- **high:** the garment state is clearly visible in at least three review
  frames and the criterion scores are stable across them.
- **medium:** temporal or wrist context resolves a partial base-camera
  occlusion without changing the apparent outcome.
- **low:** occlusion or motion blur could change at least one criterion by one
  point. Low-confidence items require an explicit adjudication note.

## Stage 1: `side_one`

The first shirt side and its sleeve should be folded inward while the untouched
side and torso remain available for the second fold.

1. **Lateral placement**
   - 2: the folded outer edge lands near the intended center strip with a
     consistent width from shoulder to hem.
   - 1: visibly over- or under-folded, but the side is fully brought inward.
   - 0: much of the side remains outside, crosses far beyond the target strip,
     or produces no stable inward fold.
2. **Sleeve and side containment**
   - 2: sleeve and side fabric are contained with at most a small tip exposed.
   - 1: a localized sleeve/edge protrusion remains.
   - 0: most of the sleeve or a large side panel remains exposed or doubled
     back.
3. **Longitudinal alignment**
   - 2: fold line is approximately parallel to the torso axis; shoulder and hem
     portions do not visibly shear in opposite directions.
   - 1: moderate taper, skew, or unequal placement along the fold.
   - 0: severe diagonal fold, twist, or displaced shoulder/hem alignment.
4. **Flatness and body preservation**
   - 2: placed layer is flat and the remaining shirt body is not displaced.
   - 1: localized ridge, wrinkle, or small displacement.
   - 0: substantial bunching, rolled fabric, or disturbance that compromises
     the next fold.

## Stage 2: `side_two`

The second side and sleeve should fold inward to create a stable, elongated
two-sided strip without undoing the first fold.

1. **Second-side placement and strip width**
   - 2: second edge lands consistently and produces an appropriately narrow,
     nearly parallel strip.
   - 1: strip is usable but visibly too wide/narrow, tapered, or offset.
   - 0: second side is substantially under/over-folded or the strip is
     unstable.
2. **Second sleeve and edge containment**
   - 2: second sleeve and side edge are contained with only a small localized
     protrusion.
   - 1: material protrusion or doubled edge is present but localized.
   - 0: most of the sleeve/side remains exposed or folded back out.
3. **Preservation and bilateral alignment**
   - 2: first fold remains intact and the two folded sides are aligned along
     the torso.
   - 1: first fold shifts or the two sides are moderately asymmetric.
   - 0: first fold is substantially undone, crossed, or twisted by the second.
4. **Flatness and straightness**
   - 2: elongated result lies flat with straight, stable layers.
   - 1: localized ridge, bunch, or edge waviness.
   - 0: widespread bunching, rolling, or a twisted/non-flat strip.

## Stage 3: `bottom_to_top`

The elongated shirt should fold from the bottom toward the top into a stable,
compact final package.

1. **Fold placement and coverage**
   - 2: bottom section lands at the intended height and covers the upper stack
     consistently.
   - 1: clearly completed but noticeably short, long, or offset.
   - 0: major under/over-fold, unstable placement, or incorrect topology.
2. **Rectangularity and compactness**
   - 2: final silhouette is compact and approximately rectangular.
   - 1: recognizable package with moderate wedge shape, asymmetry, or excess
     width.
   - 0: pile-like, broadly splayed, or lacking a stable compact outline.
3. **Layer and edge containment**
   - 2: sleeves, tails, and internal layers are contained with aligned edges.
   - 1: one localized protrusion or visibly uneven layer remains.
   - 0: large fabric sections protrude, unfold, or cross the package.
4. **Flatness and stability**
   - 2: package is low, flat, and remains settled after release.
   - 1: localized mound/ridge or mild spring-back.
   - 0: severe bunching, upright/rolled stack, or unstable spring-back.

## Behavior-event labels

Behavior labels are recorded independently from outcome quality.

### Grasp retry

A telemetry candidate remains a close/reopen followed by another close from the
same gripper within 1.5 seconds of the reopen. Each candidate receives one
semantic class after video review:

- `confirmed_misgrasp`: initial closure fails to secure the intended fabric or
  secures visibly wrong fabric, forcing a retry.
- `deliberate_regrasp`: fabric was secured or intentionally released and the
  next grasp is a purposeful reposition/refinement.
- `uncertain`: occlusion prevents a defensible distinction.

Only `confirmed_misgrasp` approach frames may be downweighted. Recovery after
the successful retry remains an ordinary demonstration. `deliberate_regrasp`
and `uncertain` events are neutral.

### Smoothing

Smoothing is a deliberate contact pass over an already placed layer or edge
without a new topology-changing lift. Each event should record exact half-open
retained/source frame intervals, acting arm, stage, and one of:

- `surface_pass`: contact travels across a panel to remove a ridge or wrinkle;
- `edge_alignment`: contact follows or nudges an edge into alignment;
- `compression_settle`: local press/hold intended to settle stacked layers;
- `uncertain_contact_refinement`: likely refinement, but occlusion prevents a
  reliable subtype.

Smoothing remains neutral for priority and reward targets. The event labels are
for diagnostics, stratified evaluation, and possible future behavior
conditioning—not evidence of low-quality action.

## Use with dense SARM reward

The v2 stage-quality label and SARM dense progress are complementary rather
than interchangeable:

- quality is a stage-level visible-outcome judgment and controls only which
  training frames are sampled more often;
- SARM predicts within-episode task progress and controls only the native
  RA-BC loss weight for the next 30-frame action chunk; and
- neither signal is applied to validation or test loss.

The rescore changed no stage boundary, retained frame, or split assignment.
Consequently, the completed SARM v1 checkpoint and stride-1 progress predictions
remain valid. The v2 SARM contract and integration audit rebind those unchanged
dense targets to this manifest's SHA and recompute every quality-stratified
diagnostic. The combined pre-launch audit is
`configs/training/quality_sarm_audits/openarm2_shirt_fold_quality_sarm_v2.json`.

This integrated objective is intentionally not a causal ablation. Its purpose
is to screen the intended training architecture without running another plain
BC recipe that is already below the acceptable behavior threshold.
