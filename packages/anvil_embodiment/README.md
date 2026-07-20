# anvil-embodiment

Offline-first, fail-closed embodiment bridges around frozen LeRobot policies.
The package keeps hardware conventions and kinematics explicit, and trains only
a bounded residual action adapter.

The canonical shirt-fold adapter maps the modified OpenArm v1 policy's degree
features through TCP FK/IK into Anvil OpenArm 2.0 radians. It pins policy
processors and kinematic models by hash, uses the target dataset's
−0.003-rad closed / +0.050-rad open gripper command calibration, keeps a
0.005-rad arm-joint command margin inside the nominal limits, and refuses live
execution until an artifact is explicitly approved.

Residual training can stream validation loss and bridge-relative joint, shoulder,
TCP, motion, and correction-saturation metrics to W&B. It also records the pinned
cache/manifest and final offline evaluation as immutable run artifacts.
