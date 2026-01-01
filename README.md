# Roadwork Zone Detection and Geo-localization (DeepLabV3+)

A semantic segmentation system designed to detect and localize roadwork zone elements (Barriers and Road Beacons) using a custom DeepLabV3+ implementation.

> **Attribution:** This project is adapted from the [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) repository by VainF.

## Project Overview

This system utilizes the DeepLabV3+ architecture with a MobileNetV2 backbone for efficient real-time segmentation of roadwork environments.

### Key Features
*   **Custom Training Pipeline:** Adapted for 3-class segmentation (Background, Barrier, Road Beacon).
*   **Real-time Monitoring:** Integrated TensorBoard logging for loss, accuracy, and mIoU tracking.
*   **Enhanced Visualization:** Custom prediction pipeline with:
    *   Class-specific colormap (Red=Barrier, Yellow=Beacon).
    *   Alpha-blended overlays for visual verification.
    *   Automatic resizing to match original input resolution.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Hitesh94-falcon/Roadwork-Zone-Detection-and-Geo-localization.git
    cd Roadwork-Zone-Detection-and-Geo-localization
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
To train the model on your custom dataset:
```bash
python train.py
```
*   **Checkpoints:** Saved in `./checkpoints` (Best model + every 10 epochs).
*   **Logs:** View real-time metrics with `tensorboard --logdir=./runs`.

### Prediction
To run inference on a folder of test images:
```bash
python predict.py
```
*   **Output:** Results (Masks + Overlays) are saved in `./results`.
*   **Configuration:** Adjust `PredictConfig` in `predict.py` to change input directories or model settings.
