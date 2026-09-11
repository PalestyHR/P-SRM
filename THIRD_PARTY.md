# Third-party acknowledgements

| Component | Source and role |
| --- | --- |
| TAP-Net, Online TAPIR, TAPNext++, TAP-Vid | [Google DeepMind TAPNet](https://github.com/google-deepmind/tapnet). Native models, dataset protocols and TAP-Vid metric equations. The repository retains its upstream Apache-2.0 terms. |
| TrackNetV3 | [Original repository](https://github.com/qaz812345/TrackNetV3). Native checkpoint, eight-frame graph, preprocessing and corrected Shuttlecock Test labels. |
| TrackNetV2 | [Preserved original implementation](https://github.com/tamaki-lab/2024_05_ding_TrackNetv2). Native model906_30 graph, weights and decoder. |
| TrackNetV1 | [Original project mirror](https://github.com/nyck33/TrackNetMirror). Model II-prime graph/weights and native Hough decoder. Refer to the original authors' terms identified by that mirror. |
| OpenCV KCF | [OpenCV-contrib 4.10.0 tracking](https://github.com/opencv/opencv_contrib/tree/4.10.0/modules/tracking). The included patch exposes same-update evidence; OpenCV retains its Apache-2.0 terms. |
| LibAUC | [LibAUC](https://github.com/Optimization-AI/LibAUC). One-way partial-AUC loss, SOPAs optimizer and DualSampler. Imported as a dependency. |
| RacketVision | [Dataset repository](https://huggingface.co/datasets/linfeng302/RacketVision). Original video/label source. |
| OTB | [Object Tracking Benchmark](http://cvlab.hanyang.ac.kr/tracker_benchmark/). Original images and bounding boxes. The visibility file in this repository is a separately described supplement. |

The TrackNetV1/V2 PyTorch graph interpreters execute user-supplied original parameters; they are not newly trained native models. The supplied V1 architecture description is derived from the released original graph. Preserve the native authors' applicable terms when using these components.

No native checkpoint, original dataset image/video, or complete third-party source tree is redistributed. Download those materials from their sources. The P-SRM code and model package is currently a private review copy, with public licensing to be assigned by the authors.
