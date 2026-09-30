# B-net: defect depth estimation from thermal B-scans

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
├── Experimental_data_test.ipynb           # evaluation of a trained model on the test set
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

## Data

The experiments use the open-source dataset described in **[dataset
reference / DOI to be added]**.

All scripts are run from the repository root and expect the data in the
following layout (the folder `open_source_dataset/` is ignored by git):

```
open_source_dataset/
├── training/
│   ├── *.npz            # one file per sequence: 'data' [T, H, W] (ΔT), 'mask' [H, W] (depth)
│   ├── data_bscans/     # created in step 2
│   └── data_masks/      # created in step 2
├── validation/          # same structure
├── testing/             # same structure
└── trained_models/      # created by the training script
```

In each `.npz` file, `data` is the temperature rise above the initial
temperature and `mask` is the defect depth map, normalised to the specimen
thickness (0 = sound material).

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
  title   = {[Paper title]},
  author  = {Jaworski, Konrad and others},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {[to be added]}
}
```

## License

This project is released under the MIT License, see [LICENSE](LICENSE).