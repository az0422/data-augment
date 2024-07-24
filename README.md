# Data Augmentation Library for Python

## Modules
 * `modules.classify.Augmentation`
 Augmenting data for classify

 * `modules.segment.Augmentation`
 Augmenting data for segmentation

## How to use
 `python3 augment.py cfg.yaml`

 Content of cfg.yaml

### File path and mode
 * `mode: segment or classify`
 * `image_path: path/to/images`
 * `label_path: path/to/labels`
 * `image_export: export/to/path/to/images`
 * `label_export: export/to/path/to/labels`

### Function setup
 * `batch_size: 32`
 * `epochs: 1024`
 * `resize: 640`

### Augmentation parameters
 * `flip_vertical: 0.5 # probability (0 - 1)`
 * `flip_horizontal: 0.5 # probability(0 - 1)`
 * `rotate_degree: 0 # degree (0 - 180)`
 * `rotate_prob: 0.5 # probability (0 - 1)`
 * `brightness_range_add: [1, 1] # brightness * mul * ratio + add, (0 - 2)`
 * `brightness_range_mul: [1, 1] # brightness * mul * ratio + add, (0 - 2)`
 * `brightness_range_ratio: [1, 1] # brightness * mul * ratio + add, (0 - 2)`
 * `noise_opacity_range: [0, 0] # noise opacity (0 - 1)`
 * `translate_vertical_range: [0, 0] # (x, y + translate) (-1 - 1)`
 * `translate_horizontal_range: [0, 0] # (x + translate, y) (-1 - 1)`
 * `rescale_ratio_range: [1, 1] # (width * resize, height * resize) (0 - 1)`

## TODO list
 * Implement for examples
