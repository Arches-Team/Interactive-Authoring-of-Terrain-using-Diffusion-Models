import os
import json

import numpy as np
import torch
from ...core.utils import array_to_image, load_image
from ...labelling.encoding import GlobalTerrainStyle

def preprocess_image(img):
    img = 2 * img - 1

    if img.ndim == 2:  # (h, w) to (h, w, 1)
        img = np.expand_dims(img, axis=2)

    img = np.transpose(img, (2, 0, 1))  # (h, w, c) -> (c, h, w)

    return torch.from_numpy(img).unsqueeze(0)


def tensor_to_img(x, bit_depth):
    x = x.cpu().numpy()
    x = np.transpose(x, (1, 2, 0))  # (c, h, w) -> (h, w, c)
    return array_to_image(x, bit_depth=bit_depth)


def load_tile(
    tile_dir,
    metadata_file='metadata.json',
    style_file='style.png',
    sketch_file='sketch.png',
    num_samples=1,
):

    with open(os.path.join(tile_dir, metadata_file)) as fp:
        metadata = json.load(fp)

    desired_resolution = metadata['resolution']
    desired_range = metadata['range']

    # Load sketch
    sketch_image = load_image(
        os.path.join(tile_dir, sketch_file),
        convert_to='RGBA'
    )
    sketch_tensor = preprocess_image(sketch_image)

    # Load style
    style_image = load_image(os.path.join(tile_dir, style_file))
    style_tensor = preprocess_image(style_image)
    

    # NOTE: Pipeline accepts batches. Expand based on num_samples
    # (c, h, w) -> (b, c, h, w)
    sketch_tensor = torch.cat([sketch_tensor] * num_samples)
    style_tensor = torch.cat([style_tensor] * num_samples)
    desired_range = [desired_range] * num_samples
    desired_resolution = [desired_resolution] * num_samples

    # Create style
    style = GlobalTerrainStyle(
        terrains=style_tensor,
        ranges=desired_range,
        resolutions=desired_resolution
    )

    return dict(
        sketch=sketch_tensor,
        style=style,
        resolution=desired_resolution,
        range=desired_range,
    )