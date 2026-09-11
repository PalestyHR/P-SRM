# Data and native checkpoints

Original media and native model checkpoints are downloaded separately. Use the group lists under `configs/splits/`; do not substitute a new random split.

## Sources

- TAP-Net, Online TAPIR, TAPNext++, TAP-Vid Kinetics and TAP-Vid DAVIS: [Google DeepMind TAPNet](https://github.com/google-deepmind/tapnet), including its checkpoint and dataset instructions.
- TrackNetV3 and Shuttlecock: [TrackNetV3](https://github.com/qaz812345/TrackNetV3). Use the official corrected Test labels when evaluating Shuttlecock Test.
- RacketVision: [dataset repository](https://huggingface.co/datasets/linfeng302/RacketVision). The experiment uses its Train and Validation groups, not Test.
- TrackNetV2 original model `model906_30`: [preserved original implementation](https://github.com/tamaki-lab/2024_05_ding_TrackNetv2), with its upstream checkpoint link.
- TrackNetV1 Model II-prime `model.3`: [original project mirror](https://github.com/nyck33/TrackNetMirror). This mirror identifies the original authors' project and terms.
- OTB2013: [OTB benchmark](http://cvlab.hanyang.ac.kr/tracker_benchmark/). The OTB2013 sequences can also be selected from an OTB2015 download. Retain the original ground-truth boxes and image filenames.

## TAP-Net / Online TAPIR

Use a separate Linux CUDA environment for the JAX native trackers. Install the upstream TAPNet dependencies and the package's `points` dependencies. The reference TAPNet checkout for the point exporters is `c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`.

```bash
git clone https://github.com/google-deepmind/tapnet.git third_party/tapnet
git -C third_party/tapnet checkout c2cbab81cc06092b5f05bfe2da7bfec54e2079c9
python -m pip install -e third_party/tapnet
python -m pip install -r requirements-points.txt
```

Follow upstream instructions to install the JAX CUDA plugin appropriate for your environment. Kinetics input has `tapvid_kinetics_development.csv` and `pickle_development_10/*.pkl` under its data root. Download checkpoints from upstream: TAP-Net's `checkpoint.npy` and Online TAPIR's `causal_tapir_checkpoint.npy`.

TAP-Net training-source export:

```bash
python -m psrm.hosts.tapnet --data-root datasets/tapvid_kinetics --design-ids configs/splits/tapnet_train.csv --checkpoint native_models/tapnet/checkpoint.npy --output-root data/tapnet/raw_train --export-evidence --export-history
python -m psrm.hosts.prepare_points --host tapnet --raw data/tapnet/raw_train --split configs/splits/tapnet_train.csv --output data/tapnet/train
```

Repeat with `tapnet_evaluation.csv`, `raw_evaluation` and `evaluation` for the evaluation partition.

Online TAPIR training-source export:

```bash
python -m psrm.hosts.tapir --data-root datasets/tapvid_kinetics --ids configs/splits/online-tapir_train.csv --checkpoint native_models/causal_tapir_checkpoint.npy --output-root data/online-tapir/raw_train
python -m psrm.hosts.prepare_points --host online-tapir --raw data/online-tapir/raw_train --split configs/splits/online-tapir_train.csv --output data/online-tapir/train
```

Repeat with the evaluation split and corresponding output paths. Native metrics and evidence are exported from the same native forward pass.

## TAPNext++

Follow the TAPNet `tapnextpp/votsp2026` checkpoint instructions and install its PyTorch dependencies. The source is Kinetics Design; the evaluation set is TAP-Vid DAVIS.

```bash
python -m psrm.hosts.tapnext --split source --data-root datasets/tapvid_kinetics --tapnet-root third_party/tapnet --folds configs/splits/tapnext_train.csv --checkpoint native_models/tapnextpp_ckpt.pt --output-root data/tapnext/raw_train
python -m psrm.hosts.prepare_points --host tapnext --raw data/tapnext/raw_train --split configs/splits/tapnext_train.csv --output data/tapnext/train
python -m psrm.hosts.tapnext --split davis --data-root datasets/tapvid_kinetics --davis datasets/tapvid_davis/tapvid_davis.pkl --tapnet-root third_party/tapnet --folds configs/splits/tapnext_train.csv --checkpoint native_models/tapnextpp_ckpt.pt --output-root data/tapnext/raw_evaluation
python -m psrm.hosts.prepare_points --host tapnext --raw data/tapnext/raw_evaluation --split configs/splits/tapnext_evaluation.csv --output data/tapnext/evaluation
```

## TrackNet datasets

For Shuttlecock Train, the root contains `Professional/match*/video` and `Amateur/match*/video`, with corresponding `csv` directories. For Test, pass the `Test` directory and supply the corrected-label directory from TrackNetV3. For RacketVision, the data root contains `<sport>/videos/`, `<sport>/all/<match>/csv/`, `<sport>/all/<match>/median.npz` and the dataset's split metadata.

Build local manifests. They point to your files and are not part of the published repository:

```bash
python -m psrm.hosts.make_manifest --dataset shuttlecock --root datasets/Shuttlecock/Train --split configs/splits/tracknetv3_shuttlecock_train.csv --output data/tracknetv3_shuttlecock/train_videos.csv
python -m psrm.hosts.make_manifest --dataset shuttlecock --root datasets/Shuttlecock/Test --corrected-test-labels third_party/TrackNetV3/corrected_test_label --split configs/splits/tracknetv3_shuttlecock_evaluation.csv --output data/tracknetv3_shuttlecock/evaluation_videos.csv
```

For RacketVision replace the dataset, root, split and output paths. For V1/V2 use their corresponding published splits. The manifest builder prepares match medians for Shuttlecock. RacketVision uses its provided medians and original 1920×1080 label coordinate system.

TrackNetV3 uses the released eight-frame, background-concatenation model and its last-frame output:

```bash
python -m psrm.hosts.tracknet --version 3 --manifest data/tracknetv3_shuttlecock/train_videos.csv --split configs/splits/tracknetv3_shuttlecock_train.csv --tracknet-root third_party/TrackNetV3 --checkpoint native_models/TrackNet_best.pt --output data/tracknetv3_shuttlecock/train --device cuda
```

Repeat for evaluation. Use the matching manifest and split paths. The exporter keeps the original warm-up-frame exclusion.

### TrackNetV1 / TrackNetV2

Convert the original Keras weights once in a Python 3.10 environment with TensorFlow/Keras 2.15 and NumPy 1.x. This produces only numerical weights and the original graph description; it does not train or alter the graph.

```bash
python scripts/convert_keras.py --version 1 --architecture adapters/tracknet/v1_model.json --checkpoint native_models/model.3 --output native_models/v1
python scripts/convert_keras.py --version 2 --checkpoint native_models/model906_30 --output native_models/v2
```

Back in the PyTorch P-SRM environment:

```bash
python -m psrm.hosts.tracknet --version 1 --manifest data/tracknetv1_shuttlecock/train_videos.csv --split configs/splits/tracknetv1_shuttlecock_train.csv --model-config native_models/v1/model.json --checkpoint native_models/v1/weights.npz --output data/tracknetv1_shuttlecock/train --device cuda
```

Use `--version 2` and V2 paths for TrackNetV2; repeat for evaluation and the RacketVision configurations. V1 retains newest-first, three-frame BGR 0–255 input and the original Hough-circle decoder. V2 retains non-overlapping RGB triplets, its original preprocessing and three output heatmaps. Their graph interpreters preserve the stored BatchNorm axis and epsilon.

## KCF / OTB2013

Build the small observer against OpenCV and OpenCV-contrib 4.10.0. The patch exposes the response and candidate before the native rejection decision; it does not alter the decision or tracker update.

```bash
git clone --branch 4.10.0 --depth 1 https://github.com/opencv/opencv.git third_party/opencv
git clone --branch 4.10.0 --depth 1 https://github.com/opencv/opencv_contrib.git third_party/opencv_contrib
git -C third_party/opencv_contrib apply --ignore-space-change ../../adapters/kcf/observer.patch
cmake -S third_party/opencv -B build/opencv -DOPENCV_EXTRA_MODULES_PATH=$PWD/third_party/opencv_contrib/modules -DBUILD_LIST=core,imgproc,imgcodecs,video,tracking -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF
cmake --build build/opencv --config Release --parallel
cmake -S adapters/kcf -B build/kcf -DOpenCV_DIR=$PWD/build/opencv
cmake --build build/kcf --config Release --parallel
```

On Windows, use the corresponding absolute paths for CMake variables and the `build/kcf/Release/kcf_check.exe` executable. Ensure the built OpenCV libraries are on the loader's path.

```bash
python -m psrm.hosts.kcf --otb-root datasets/OTB2015 --labels annotations/otb2013/visibility.csv --split configs/splits/kcf_train.csv --observer build/kcf/kcf_check --output data/kcf/train
```

The exporter applies the OTB2013 frame ranges, converts one-based box coordinates to zero-based, joins the released visibility labels, and groups targets from the same video in the same fold. It does not relabel the data. Then follow the KCF commands in [reproduce.md](reproduce.md).
