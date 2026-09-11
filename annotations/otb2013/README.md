# OTB2013 supplementary visibility annotations

Initial annotations were generated using **Qwen3-VL-2B-Instruct** (Qwen/Qwen3-VL-2B-Instruct). The annotations were corrected and passed human spot-check review on a subset of cases.

These are supplementary annotations associated with P-SRM, separate from the official OTB bounding-box annotations. The original images are not included.

## Files and indexing

- [visibility.csv](visibility.csv): 29,491 target-frame records across 51 target sequences and 50 video groups.
- [prompt.txt](prompt.txt): the exact prompt used for initial annotation.

The frame_index field is zero-based within the selected target sequence. Use image_path, relative to the dataset root, to resolve the original image filename. Do not assume that frame_index + 1 equals the image filename number. The two Jogging targets share one video_group; keep them together when constructing grouped folds.

Visibility 1 means that at least part of the tracked target is physically visible, including partial occlusion. Visibility 0 means complete occlusion or absence from the view. The initial annotation prompt also allowed -1 for uncertainty; the released labels are resolved binary labels.

## Annotation input

Image 1 is the first-frame target reference crop. Image 2 is the current frame's ground-truth target region with 20% context padding. Aspect ratio is preserved and each crop is padded to 256 x 256 pixels. No tracker response, candidate prediction, confidence score, or candidate-correctness label is supplied to the annotator. Visibility is not propagated between frames.

Generation used greedy decoding, a maximum of 8 output tokens, batch size 32, BF16, and the model's SDPA attention implementation. The actual prompt is provided verbatim in prompt.txt.

These labels are supervision/evaluation data, not runtime input to the P-SRM recovery module.
