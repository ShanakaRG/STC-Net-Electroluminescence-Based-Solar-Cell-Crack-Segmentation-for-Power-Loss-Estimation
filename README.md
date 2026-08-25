# SolarTopoCrackNet

A Python 3.6 / Torch 1.7 compatible package for EL solar-cell defect segmentation.

## Main novelty

SolarTopoCrackNet extends the earlier edge-guided baseline with four task-specific ideas:

1. **Frequency-guided spectral pyramid** to suppress regular solar-cell texture and highlight abnormal EL responses.
2. **Orientation-aware crack mixer** to model horizontal, vertical, diagonal-like, and local crack patterns.
3. **Topology head** that predicts a thin skeleton / centerline map of the defect region.
4. **Boundary-topology consistency loss** that couples region, boundary, and topology predictions.


## STC-Net Architecture

![STC-Net model architecture](model.png)


This is designed for EL segmentation settings where many mistakes come from:
- thin cracks
- fragmented masks
- fuzzy boundaries
- confusion with regular cell texture

---

## Folder layout expected

The code supports any image/mask folder names as long as images and masks can be paired by filename stem.

Example:

```text
train/
  images/
  masks/
test/
  images/
  masks/
```

If you do not have a validation set, keep `val_images: null` and `val_masks: null` in the YAML. The code will split a validation set from training automatically.

---

## Key files

- `train.py` — training entry point
- `infer.py` — inference script
- `datasets.py` — dataset, augmentations, mask-edge-topology target generation
- `losses.py` — segmentation, edge, topology, and consistency losses
- `models/solar_topo_crack.py` — main proposed model
- `configs/solar_topo_crack_py36.yaml` — ready config for Python 3.6 / Torch 1.7

---

## Environment

This package is written to match:
- Python 3.6.13
- torch 1.7.1
- torchvision 0.8.2
- OpenCV 3.4.x

Install lightweight extras only:

```bash
pip install -r requirements_py36_torch171.txt
```

---

## Training

Edit `configs/solar_topo_crack_py36.yaml` and set:

```yaml
data:
  train_images: /path/to/train/images
  train_masks: /path/to/train/masks
  val_images: null
  val_masks: null
  test_images: /path/to/test/images
  test_masks: /path/to/test/masks
```

Then run:

```bash
python train.py --config configs/solar_topo_crack_py36.yaml
```

If you have multiple GPUs visible, the script can use `DataParallel` automatically.

---

## Inference

```bash
python infer.py \
  --config configs/solar_topo_crack_py36.yaml \
  --checkpoint outputs/solar_topo_crack_py36/checkpoints/best.pt \
  --input_dir /path/to/test/images \
  --output_dir ./predictions \
  --save_aux
```


CUDA_VISIBLE_DEVICES=2 python infer_crack_power_eval.py \
  --config configs/solar_edge_msf_binary_py36.yaml \
  --checkpoint outputs/solar_edge_msf_binary_py36/checkpoints/best.pt \
  --input_dir D:\Self_reaserch\solar panel\codes\data\PVEL_S\test\defect \
  --gt_mask_dir D:\Self_reaserch\solar panel\codes\data\PVEL_S\test\label \
  --output_dir ./outputs/test_infer_power \
  --nominal_power 1.0 \
  --cell_mode full_image \
  --threshold 0.4



This saves:
- predicted binary masks
- overlay images
- optional edge probability maps
- optional topology probability maps

---

## Notes

- For binary segmentation, masks should be black background and white foreground.
- `threshold: 0.5` is a starting point only. You should sweep threshold on the validation set.
- If the dataset is small, try `image_size: 640` and reduce `batch_size`.
- If the masks are very sparse, increase `seg_pos_weight` and `topo_pos_weight`.


## v2.1 notes
This package uses the original working SolarTopoCrack architecture with lightweight SE gating in residual blocks, OHEM BCE+Dice, topology warmup, automatic validation-threshold search, and optional flip-TTA at inference.
