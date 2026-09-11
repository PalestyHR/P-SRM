# Host interface

P-SRM consumes evidence for the fixed candidate that the native host rejected. It returns a new validity mask, while keeping coordinates and native acceptances unchanged. Keep the host state independent of recovery.

## Inputs

| Input | Shape | Meaning |
| --- | --- | --- |
| `positions` | `[N,2]` or `[N,4]` | Native candidate points or boxes, including accepted rows |
| `native_valid` | `[N]` | The host's own Boolean decision |
| `dense_evidence` | `[R,3,32,32]` | Task evidence, candidate support, valid support |
| `candidate_metadata` | `[R,4]` | Normalized position, in-frame flag, boundary flag |
| `history` | `[R,21]` | Causal features derived only from native outputs |
| `native_margin` | `[R]` | Host-specific decision evidence |

`R` is the number of native rejections. All four model-input arrays follow `positions[~native_valid]`. The network does not consume ground truth, future states or labels.

Use `psrm.data.build_dense_projection` for TAP-Net/Online TAPIR. The host exporters implement the local evidence projection for TrackNet, the candidate-centered axis evidence for TAPNext++, and the box-support projection for KCF. Keep these host conventions: an arbitrary heatmap normalization is not equivalent to the trained input.

## Causal history

Create one `CausalHistory(query_frame)` per query. For each subsequent native output in time order, call:

```python
row = state.update(frame_index, candidate_xy, native_margin,
                   native_visible_score, native_valid)
```

A native acceptance updates the trusted state and returns `None`. A rejection returns its 21-D feature vector. Never feed a recovery decision back as a native acceptance. For KCF, use fixed box centers in native pixel coordinates and omit unscored initialization/bootstrap states. The KCF exporter handles this convention.

Point-host coordinates are in the 256×256 model domain. TrackNet history uses the 512×288 evaluation domain; V1's 640×360 intensity-grid candidate is mapped mechanically by 0.8. KCF history uses native image pixels. Do not interchange model bundles between these domains.

## Native decision evidence

TAP-Net uses its occlusion decision; Online TAPIR uses its native visibility score; TAPNext++ uses its visibility logit. TrackNetV2/V3 use the 0.5 response threshold. TrackNetV1 retains its original intensity-threshold and Hough-circle decoder: its supplied margin is `peak / 255 - 127 / 255`, an intensity-threshold prior rather than a signed margin for the entire Hough decision. KCF uses the original response peak minus its native threshold.

A recovered coordinate is the same fixed candidate supplied to P-SRM. Recovery does not repair a localization error.
