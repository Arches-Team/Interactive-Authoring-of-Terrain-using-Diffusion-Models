# Interactive Authoring of Terrain using Diffusion Models

[Website](https://simonperche.github.io/Interactive-Authoring-of-Terrain-using-Diffusion-Models/) | [Paper](https://hal.science/hal-04324336/document) | [Video](https://www.youtube.com/watch?v=KyN9O9m5HS0) 

## Usage

Install virtual environment and dependencies:
```sh
make install
```

---
## Contents

- [Install developer requirements](#install-developer-requirements)
- [Data collection](#data-collection)
    - [Masking](#masking)
    - [Downloading](#downloading)
    - [Preprocessing](#preprocessing)
    - [Dataset preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Blender add-on](#blender-add-on)
    
---

Note: You can download our sketch-to-terrain trained model from [this Google Drive Folder](https://drive.google.com/file/d/1t9BWJWPUnsETadSkTdSmcrHXPQYka5zK/view?usp=drive_link).

## Install developer requirements
Install additional packages that are needed to do data collection, training, and evaluation.
```sh
make install-dev
```

Next, enable the virtual environment with `source venv/Scripts/activate` if you are on Windows, and `source venv/bin/activate` if you are on Mac/Linux.

## Data collection
This section describes how to generate your own training data.

Note: we simplify the process with `make` commands, but if you want more control (e.g., select zoom level), refer to the relevant scripts and their arguments (see the [Makefile](Makefile) for example).

### Masking

1. Generate the satellite, coverage, and user masks.
    ```sh
    make mask
    ```

2. Using an image editing software, edit the user mask (found at `data/masks/user_mask.jpg`) to include (white) or exclude (black) different regions. You can use the satellite image as a guide for selecting regions.


### Downloading

Download the tiles that are in both the user mask and coverage mask.
    ```sh
    make download
    ```

Valid elevation tiles are saved to `data/elevation/valid` in the form `<zoom_level>_<tile_y>_<tile_x>.npz` and satellite images are saved to `data/satellite/raw` in the form `<zoom_level>_<tile_y>_<tile_x>.jpg`.
Please note that this process will download **a lot** of data.

### Dataset preparation
This section describes how to generate the different .tar.gz datasets.

1. Generate dataset (including elevation, derivative, satellite, and sketch images).
    ```sh
    make dataset
    ```

    This saves the dataset to `data/processed/to_divide/<zoom_level>/<factor>`. Tiles are stored as folders, with subfolders containing the elevation, derivative, satellite, and sketch data.
    
2. Next, spread the tiles across the `train`, `valid`, and `test` folders. Remember to place these tiles in a `<zoom_level>/<factor>` folder within `train`, `valid`, or `test`. We recommend a random split of: train (90%), valid (5%), test (5%).

3. (Optional - only run if you want to share the datasets) Generate the archives:
    ```sh
    make archives
    ```

    This will save several .tar.gz files to the `data/datasets` folder.

## Training
This section describes how to train the various diffusion models. In each case, we split the types of arguments into *model*, *input/output*, and *training* flags. As mentioned in the paper, we use the same model flags in each case (these are also the defaults, so you can ignore this if you don't need to change anything):
```sh
MODEL_FLAGS="--image_size 256 --inner_channel 64 --num_res_blocks 3 --attention_resolutions 16 --channel_mults 1 2 4 8"
```


The training arguments also (for the most part) stay the same. The only change we make is use a learning rate of 1e-4 for unconditional training and 5e-5 for conditional training. Feel free to adjust these depending on your training architecture.

```sh
TRAINING_ARGS="--batch_size 4 --num_epochs 50 --amp --num_workers 2 --sample_seed 0 --num_samples 8 --log_steps 5000 --eval_steps 5000 --sample_steps 5000 --save_steps 5000 --save_limit 5"
```

### Unconditional Models

- Elevation
    ```sh
    IO_FLAGS="--target elevation/256x256.png --input_image_channels 3 --output_image_channels 1 --model_dir models/diffusion/unconditional-elevation"
    ```

- Satellite
    ```sh
    IO_FLAGS="--target satellite/1024x1024.jpg --input_image_channels 3 --output_image_channels 3 --model_dir models/diffusion/unconditional-satellite"
    ```

Then run:

```sh
python -m terrain_diffusion.models.diffusion.train --lr 1e-4 $MODEL_FLAGS $TRAINING_ARGS $IO_FLAGS
```

### Conditional Models

- Sketch-to-terrain
    - Elevation
        ```sh
        IO_FLAGS="--cond_image sketch-256x256.png --cond_image_channels 4 --target elevation/256x256.png --input_image_channels 3 --output_image_channels 1 --model_dir models/diffusion/sketch-to-elevation"
        ```

Then run:
```sh
python -m terrain_diffusion.models.diffusion.train --lr 5e-5 $MODEL_FLAGS $TRAINING_ARGS $IO_FLAGS
```

## Evaluation
After training/downloading the models, move them to the `models/diffusion/final` folder, with the names corresponding to those used in the `TASK_MODEL_DEFAULTS` dictionary in `src/evaluation/eval.py`. This also requires that the test data is located in the `data/processed/test` folder. 

To evaluate on a specific task, use `--task <name_of_task>` where `<name_of_task>` is one of: `unconditional-elevation`, `unconditional-derivative`, `unconditional-satellite`,
`sketch-to-elevation`, `sketch-to-derivative`,
`upscale-elevation-32`, `upscale-derivative-32`,
`uncropping-elevation`, `uncropping-derivative`,
`elevation-to-satellite`, or `derivative-to-satellite`

To change the number of sampling timesteps, use `--sampling_timesteps <num>`, where `<num>` is an integer between 1 and 1000 (250 is recommended).
Then run:

```sh
python -m src.evaluation.eval --task <name_of_task> --generate --sampling_timesteps <num> --evaluate
```

This will create a folder `data/evaluation/<task_name>_<num>` with all the input, generated, and target images (in the case of conditional models) or just generated data (in the case of unconditional models).

## Inference
To perform inference, simply load the model (e.g., using `torch.load`) and pass an image (stored as a numpy array with values between 0 and 1) through it.
```python
import numpy as np
from src.models.diffusion.inference import InferenceModel

# Make sure the diffusion model class is visible in the path (needed for unpickling)
import sys
sys.path.append('src/models/diffusion') # noqa

# Load the model
path = 'models/diffusion/final/sketching_elevation.pt'
model = InferenceModel(path, normalise_output=True)

# Get an input image in the form (w, h, c) and of type np.float32
# (Random tensor of correct shape shown here)
input_image = np.random.rand(256, 256, 3).astype(np.float32)

# Set sampling options
num_samples = 1
sampling_timesteps = 100

# Generate a list of images (of type List[PIL.Image])
list_of_images = model(
    img=input_image,
    num_samples=num_samples,
    sampling_timesteps=sampling_timesteps
)
print(list_of_images)
```

## Blender add-on

This repository contains a Blender add-on.

### Installation

1.  Download the required [`sketch-to-terrain` model](https://drive.google.com/file/d/1t9BWJWPUnsETadSkTdSmcrHXPQYka5zK/view?usp=drive_link).
and place it in the `models/diffusion` directory.
2.  Zip the entire project (including the `models` directory) or use ```make blender-addon```.
3.  In Blender, go to `Edit > Preferences > Add-ons`.
4.  Click `Install` and select the zipped project file.
5.  Enable the add-on.

### Included Files

*   `app/terrain_dm.blend`: A sample Blender file demonstrating the setup with the add-on and pre-configured terrain data/meshes.

### Notes

*   Ensure the `models/diffusion` directory is correctly set up before zipping.


## Citation
If you find our work useful, please consider citing:
    
```
@article{lochner2023terraindiffusion,
author = {Lochner, J. and Gain, J. and Perche, S. and Peytavie, A. and Galin, E. and Guérin, E.},
title = {Interactive Authoring of Terrain using Diffusion Models},
journal = {Computer Graphics Forum},
doi = {https://doi.org/10.1111/cgf.14941},
year = {2023}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

