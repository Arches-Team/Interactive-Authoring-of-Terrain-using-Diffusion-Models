import bpy

import os
import threading
import time

import numpy as np

from .. import utils
from ..operators.style import replace_style

from ...terrain_diffusion.models.diffusion.model import TerrainDiffusionPipeline
from ...terrain_diffusion.models.diffusion.image_utils import preprocess_image
from ...terrain_diffusion.labelling.encoding import GlobalTerrainStyle


# # Load the model
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print('Load sketch model')
sketch_model_path = utils.get_filepath_in_package(
    '../models/diffusion/sketch-to-terrain/final')

sketch_model = TerrainDiffusionPipeline.from_pretrained(
    sketch_model_path).to(device)


class InterruptableGenerator:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_generator = []

        self.current_owner = None

        self._is_running = False

    @property
    def is_running(self):
        with self.lock:
            return self._is_running

    def update_generator(self, new_generator):
        # handle incoming request to update generator

        with self.lock:
            was_running = self._is_running

            self._is_running = True
            self.current_owner = threading.current_thread()

            old_generator = self.current_generator
            self.current_generator = new_generator

        # TODO dispose of previous generator necessary?
        del old_generator

        return was_running

    def stop(self):
        # update with empty generator
        return self.update_generator(iter(()))

    def __iter__(self):
        return self

    def __next__(self):
        # Only one thread may get next item at one time
        with self.lock:
            if threading.current_thread() is self.current_owner:  # Small optimisation
                try:
                    return next(self.current_generator)
                except StopIteration:
                    # Owner thread has finished generating, meaning the generator is empty now
                    self._is_running = False
                    raise  # Reraise exception

            else:
                raise StopIteration


sketch_generator = InterruptableGenerator()


def generate_terrain(use_fp16=True, **kwargs):
    with torch.autocast(device_type=device.type, enabled=use_fp16):
        yield from sketch_model(**kwargs)


def handle_new_input():
    # TODO find a way to stop other threads?
    generation_props = bpy.context.scene.generation_props
    display_settings_props = bpy.context.scene.display_settings_props

    # TODO: report if input_sketch is empty
    sketch_bpy = bpy.data.images[generation_props.input_sketch]
    sketch = utils.to_np_array(sketch_bpy, grayscale=False).astype(np.float32)
    sketch_data = utils.rgb_to_rgba(sketch)

    if not 'style' in bpy.data.images or list(bpy.data.images['style'].size) == [0, 0]:
        replace_style(bpy.context, 'mountains')

    style_bpy = bpy.data.images['style']
    style_data = utils.to_np_array(
        style_bpy, grayscale=True).astype(np.float32)

    num_samples = generation_props.nb_sample

    # NOTE: Pipeline accepts batches.
    # (c, h, w) -> (b, c, h, w)
    cond_tensor = preprocess_image(sketch_data).to(device)
    target_tensor = preprocess_image(style_data).to(device)

    cond_tensor = cond_tensor.expand(num_samples, -1, -1, -1)
    target_tensor = target_tensor.expand(num_samples, -1, -1, -1)
    desired_range = [generation_props.terrain_range] * num_samples
    desired_resolution = [generation_props.terrain_resolution] * num_samples

    # Create style
    style = GlobalTerrainStyle(
        terrains=target_tensor,
        ranges=desired_range,
        resolutions=desired_resolution
    )

    seed = generation_props.seed
    if generation_props.random_seed:
        seed = utils.random_max_int()
        generation_props.seed = seed

    # Return model parameters:
    model_kwargs = dict(
        cond_image=cond_tensor,
        terrain_style=style,
        num_inference_steps=generation_props.sampling_steps,
        guidance_scale=generation_props.guidance,
        desired_range=desired_range,
        desired_resolution=desired_resolution,

        eta=generation_props.eta,
        seed=[generation_props.seed+i for i in range(num_samples)],
        use_fp16=generation_props.fp16,

        return_feedback=True,  # Always return generator
        # progress_bar=False
    )

    model_output = generate_terrain(**model_kwargs)

    was_running = sketch_generator.update_generator(model_output)

    latest_outputs = None
    for x in sketch_generator:
        outputs = x.images
        outputs = (outputs + 1)/2
        outputs = outputs.cpu().numpy()

        if generation_props.display_generation and (x.step >= display_settings_props.ignore_first_x_steps):
            display_outputs(outputs)

        latest_outputs = outputs

    if not generation_props.display_generation:
        # TODO do not update if not finished
        # If another stroke comes in while generating, it will update to the latest
        display_outputs(latest_outputs)


def display_outputs(outputs):
    if outputs is None:
        return

    for i, output in enumerate(outputs):
        output = np.transpose(output, (1, 2, 0))  # (c, h, w) -> (h, w, c)

        name = f'ld_sketch_sample{i}'
        utils.to_bpy_img(output, name)
        utils.update_2d_3d_views_img(name)


def start_thread_gen():
    t = utils.StoppableThread()
    t.run = handle_new_input
    t.setDaemon(True)
    t.setName('sketch_generation')
    t.start()
