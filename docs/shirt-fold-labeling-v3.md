# OpenARM2 shirt-fold semantic labeling v3

This revision separates the motion semantics that were previously collapsed
into the three outcome intervals. The versioned source artifact is
`configs/training/semantic_manifests/openarm2_shirt_fold_5stage_v1.json`.
The frozen three-stage v2 priority, SARM, and quality artifacts are historical
inputs and are not modified.

## Five-stage vocabulary

1. `side_one`: fold the first shirt side and sleeve inward.
2. `recenter_pull`: optionally translate the partly folded shirt into a
   reachable, aligned workspace. This does not advance task completion.
3. `side_two`: acquire and fold the remaining side across the centerline.
4. `strip_refinement`: optionally smooth, press, or align the two-sided strip.
5. `bottom_to_top`: acquire the lower edge and complete the final fold.

`recenter_pull` and `strip_refinement` are optional. An absent stage is encoded
as a zero-length half-open interval with `present=false`; it is never filled
with unrelated transition frames merely to make every episode look alike.

The stages form an ordered, gap-free partition after zero-length optional
stages are ignored. Grasp attempts, successful grasps, releases, and retries
remain event annotations nested under these stages rather than additional
semantic stages.

## Proposed boundary generation

The first proposal is deterministic and auditable. It reads the pinned 16-D
action trace and uses right-first gripper indices 7 and 15.

- A gripper value below 0.02 is closed.
- Simultaneous closures shorter than five frames are ignored.
- Open gaps of at most six frames are bridged so command chatter does not
  invent a new manipulation cycle.
- A distinct early cycle followed by a later fold cycle identifies
  `recenter_pull`. Its boundary is the midpoint of the largest stable
  dual-release gap.
- The initial `strip_refinement` proposal begins 75% through the final
  bimanual cycle and ends at the frozen v2 second-side outcome boundary.

The recenter proposal is supported by a separated early and late bimanual
cycle in 27 of 33 episodes. Episodes 2, 5, 10, 20, 22, and 27 have no distinct
early cycle and therefore propose `recenter_pull` as absent. Refinement onset
is deliberately marked low confidence because slow smoothing and slow fold
transport cannot be separated reliably from telemetry alone. It requires
three-camera human review.

Regenerate the manifest with:

```bash
uv run python scripts/training/build_openarm2_shirt_fold_semantic_manifest.py
```

## Outcome quality is not a motion-stage label

The 99 blinded v2 quality labels describe three garment states: after the
first side, after the complete second-side sequence, and after the final fold.
They are retained as `outcomes`, separate from the five motion stages.

In particular, the old `side_two` quality score was observed after any strip
refinement. It must not be copied onto the shortened `side_two` motion alone.
The five-stage review bundle attaches that observation to the end of
`strip_refinement` while preserving its original name and provenance.

This contract is review-gated. Existing SARM progress and priority manifests
remain three-stage artifacts until the proposed boundaries have been reviewed
and a new five-stage reward/progress artifact is emitted. Cross-version reward
and stage metrics are not directly comparable.
