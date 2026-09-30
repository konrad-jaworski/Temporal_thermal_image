# B-net: defect depth estimation from thermal B-scans

Code accompanying the paper **"Thermal B-scans enable simultaneous
reconstruction of subsurface defect depth and shape"** by K. Jaworski,
M. Sobczak and Ł. Pieczonka.

B-net estimates the depth of subsurface defects from active thermography
data. Instead of processing the full 3D thermal sequence `[T, H, W]` at once,
the sequence is cut into **thermal B-scans**: one line of the camera image
followed over time, `[T, W]`. For every B-scan the network predicts the
defect depth at each position along the line, `[W]`. Stacking the predictions
of all lines gives a depth map of the whole specimen.

Each B-net variant consists of:

1. a U-Net with an ImageNet-pretrained ResNet-34 encoder
   ([segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)),
   which turns the B-scan into a feature map,
2. a step that collapses the time axis into one feature vector per position,
3. a 1D regression head that outputs the normalised depth (0 = no defect).

## Repository structure

```
├── train_unet_bscan.py                    # training script
├── networks/
│   └── Unets.py                           # B-net architectures
├── data/
│   └── data_operators.py                  # PyTorch dataset for B-scans
├── helper_functions/
│   ├── remove_baseline.py                 # raw temperature -> temperature rise (ΔT)
│   ├── preparation_of_b_scans.py          # 3D sequences -> B-scans + depth targets
│   ├── Global_normalization_parameter_finder.py  # global normalisation constant
│   ├── helper_functions.py                # augmentations and resizing
│   ├── helper_data_generator.py           # synthetic defect layouts for simulations
│   ├── Metrics_experimental_data.py       # evaluation metrics and error plots
│   └── Thermography_tools.py              # TSR fitting, heating-stop detection
├── data_generation_scheme.ipynb           # design of the simulated defect layouts
├── Experimental_data_test.ipynb           # evaluation on the experimental (PVC) test set
├── Simulation_data_val_test.ipynb         # evaluation on the simulated validation and test sets
└── results_analysis_simulations.ipynb     # comparison of model variants (figures)
```

## Installation

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/konrad-jaworski/Temporal_thermal_image.git
cd Temporal_thermal_image
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA version first,
following the instructions at <https://pytorch.org/get-started/locally/>.

The pretrained ResNet-34 ImageNet weights are downloaded automatically by
segmentation_models_pytorch the first time a model is created (internet
access is needed once; the weights are then cached locally).

## Data and trained models

The data and the trained models are **not stored in this repository**. They
can be downloaded from **[link to be added]**. The package contains:

- **3D thermographic sequences** of both datasets, already divided into the
  training, validation and test splits used in the paper, one `.npz` file per
  recording,
- **trained model weights** (`best_model_clean.pth`) of every B-net variant,
- **training curves** (`train_log.pt`, `val_clean_log.pt`: training and
  validation MSE per epoch) and **run settings** (`run_config.pt`) of every
  model.

The B-scans used for training are not included; they are generated from the
`.npz` sequences with the scripts below (steps 2 and 3 of Usage).

In each `.npz` file, `data` is the temperature sequence `[T, H, W]` after
baseline removal (temperature rise above the initial temperature) and `mask`
is the defect depth map `[H, W]`, normalised to the specimen thickness
(0 = sound material).

### Simulated dataset (CFRP)

Prepared for this study with the FEM software MARC: 20 simulated pulse
thermography scenes of a carbon-fibre-reinforced polymer plate,
0.1 × 0.1 m and 3.5 mm thick, with circular flat-bottomed holes of varying
diameter. Defect depth is defined by the material removed from the rear side,
from 10 % to 90 % in steps of 5 %. A 5 s heating pulse (45 W halogen lamp) is
followed by 30 s of cooling, so both the heating and cooling phases are
available. The scenes are split into 10 training, 5 validation and 5 test
scenes with **non-overlapping depth levels** (validation: 25, 45, 65, 85 %;
test: 15, 35, 55, 75 %), which tests generalisation to depths not seen in
training.

### Experimental dataset (PVC)

The open-source pulsed thermography dataset of Wei et al. (*Appl. Sci.* 13,
2901 and 13, 13093, 2023): 38 recordings of 5 mm thick PVC specimens with
circular and rectangular defects (50–90 % material removed), cooling phase
only, 181 s at 10 Hz. The split into 26 training, 6 validation and 6 test
recordings follows the PT-Fusion study (Salah et al., *Sci. Rep.* 16, 12926,
2026). As in that study, the sequences were processed with eighth-order
thermographic signal reconstruction (TSR). Please cite the original dataset
papers when using these data.

### Folder layout

All scripts are run from the repository root. Unpack the downloaded data so
that each split sits in its own folder, for example for the experimental
dataset (the folder `open_source_dataset/` is ignored by git):

```
open_source_dataset/
├── training/
│   ├── *.npz            # original sequences (from the download)
│   ├── data_bscans/     # created in step 2
│   └── data_masks/      # created in step 2
├── validation/          # same structure
├── testing/             # same structure
└── trained_models/      # created by the training script
```

## Usage

The settings of every script (paths, options) are set at the bottom or top
of the file; edit them there before running.

**1. Baseline removal** (only for raw temperature recordings)

Set `input_folder` and `output_folder` in `helper_functions/remove_baseline.py`, then:

```bash
python helper_functions/remove_baseline.py
```

**2. Cutting the sequences into B-scans**

In `helper_functions/preparation_of_b_scans.py`, set the input and output
folders for each split (e.g. `open_source_dataset/training` →
`open_source_dataset/training/data_bscans` and `.../data_masks`) and the
scan direction, then run it once per split:

```bash
python helper_functions/preparation_of_b_scans.py
```

**3. Normalisation constant**

Computes the maximum temperature rise over the training set and saves it as
`normalization_params_experimental_cooling_only.npz` in the repository root:

```bash
python helper_functions/Global_normalization_parameter_finder.py
```

**4. Training**

```bash
python train_unet_bscan.py
```

The model(s) to train are chosen in the `models` / `models_names` lists in
the script. For every model, the best and last weights, the loss curves and
the run settings are written to `open_source_dataset/trained_models/<model name>/`.

**5. Evaluation**

`Experimental_data_test.ipynb` loads a trained model, predicts the depth for
all test B-scans and computes the errors with the functions in
`helper_functions/Metrics_experimental_data.py`.

## Model variants

| Class | Time-axis collapse | Regression head |
|---|---|---|
| `BnetMean` | mean over time | per position (1 × 1 convolutions) |
| `BnetSmallKernel` | learned, stacked vertical convolutions | per position |
| `BnetSmallKernelSmarter` | learned, stacked vertical convolutions | uses neighbouring positions (kernel 3) |
| `BnetSmallKernelSmarterRefine` | learned, stacked vertical convolutions | as above + residual 1D refinement |

All variants take input of shape `[B, 3, 512, 512]` and return a depth
profile `[B, 512]` in the range (0, 1).

## Citation

If you use this code, please cite:

```bibtex
@article{jaworski2026bnet,
  title   = {Thermal B-scans enable simultaneous reconstruction of subsurface defect depth and shape},
  author  = {Jaworski, Konrad and Sobczak, Micha{\l} and Pieczonka, {\L}ukasz},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {[to be added]}
}
```

## License

This project is released under the MIT License, see [LICENSE](LICENSE).