
import json
from dataclasses import dataclass, field
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


from .model import TerrainDiffusionPipeline
from .image_utils import tensor_to_img
from ...core.dataclass_argparser import CustomArgumentParser
from ...core.terrain_dataset import TerrainDataset
from ...core.terrain_transforms import UnnormaliseTransform


@dataclass
class GenerationArguments:
    model_dir: str = field(
        default='./models/diffusion/sketch-to-terrain/final',
        metadata={
            'help': 'Directory of model'
        }
    )

    seed: int = field(
        default=0,
        metadata={
            'help': 'Seed for reproducibility'
        }
    )
    timesteps: int = field(
        default=250,
        metadata={
            'help': 'Number of timesteps used for sampling'
        }
    )
    guidance_scale: float = field(
        default=1.0,
        metadata={
            'help': 'Guidance scale for classifier-free guidance'
        }
    )

    use_fp16: bool = field(
        default=False,
        metadata={
            'help': 'Run in fp16 mode'
        }
    )

    enable_attention_slicing: bool = field(
        default=False,
        metadata={
            'help': 'Whether to enable attention slicing (more memory efficient, but slower)'
        }
    )

    attention_slice_size: str = field(
        default='auto',
        metadata={
            'help': 'If `enable_attention_slicing`, use this as the slice_size'
        }
    )


@dataclass
class InferenceArguments(GenerationArguments):
    input_folder: str = field(
        default='./data/evaluation/inputs',
        metadata={
            'help': 'Path to input folder'
        }
    )
    output_folder: str = field(
        default='./data/evaluation/outputs',
        metadata={
            'help': 'Path to output folder'
        }
    )
    batch_size: int = field(
        default=1,
        metadata={
            'help': 'Number of images to generate at a time'
        }
    )

    cond_image: str = field(
        default='sketch-256x256.png',
        metadata={
            'help': 'Conditional image'
        }
    )
    target_image: str = field(
        default='elevation-256x256.png',
        metadata={
            'help': 'Target image'
        }
    )
    scale_factor: int = field(
        default=1,
        metadata={
            'help': 'Scale factor (for conditional image). Used for super-resolution'
        }
    )


def main():
    parser = CustomArgumentParser(
        (
            InferenceArguments,
        ),
        description='Run inference on a diffusion model'
    )

    inference_args, = parser.parse_args_into_dataclasses()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    output_file_name = f'output_{inference_args.seed}_{inference_args.timesteps}_{inference_args.guidance_scale}.png'

    index_mapping = dict(
        cond_image=inference_args.cond_image,
        target_image=inference_args.target_image
    )

    test_dataset = TerrainDataset(
        folder=inference_args.input_folder,
        data_augmentation=False,
        index_mapping=index_mapping,
        seed=inference_args.seed,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=inference_args.batch_size,
        shuffle=False,
        num_workers=1,
        persistent_workers=True
    )

    pipeline = TerrainDiffusionPipeline.from_pretrained(
        inference_args.model_dir)
    pipeline = pipeline.to(device)  # Move pipeline to device

    if inference_args.enable_attention_slicing:
        pipeline.enable_attention_slicing(
            slice_size=inference_args.attention_slice_size
        )

    unnormalise_transform = UnnormaliseTransform()
    to_pil = transforms.ToPILImage()

    for batch in tqdm(test_dataloader):
        target_image = batch['target_image'].to(device)
        cond_image = batch['cond_image'].to(device)

        # Resize conditional image if needed
        if inference_args.scale_factor > 1:
            cond_image = F.interpolate(
                cond_image, scale_factor=inference_args.scale_factor)

        # Specify desired terrain style
        # In this case, we want it in the style of the target image
        terrain_style = pipeline.create_terrain_style(
            target_image, batch['metadata'])

        # Generate terrain from sketch, in the style of the target terrain
        with autocast(enabled=inference_args.use_fp16):
            outputs = pipeline(
                cond_image=cond_image,
                terrain_style=terrain_style,
                num_inference_steps=inference_args.timesteps,
                guidance_scale=inference_args.guidance_scale,
                seed=inference_args.seed
            ).images

        target_image = unnormalise_transform(target_image)
        cond_image = unnormalise_transform(cond_image)
        outputs = unnormalise_transform(outputs)

        # Reshape metadata
        all_metadata = []
        for i in range(target_image.shape[0]):
            item = {}
            for k in batch['metadata']:
                item[k] = batch['metadata'][k][i].item()
            all_metadata.append(item)

        for metadata, cond_img, output_img, target_img in zip(all_metadata, cond_image, outputs, target_image):
            tile_name = f"{metadata['zoom']}_{metadata['tile_y']}_{metadata['tile_x']}"

            cond_img = to_pil(cond_img)
            output_img = tensor_to_img(output_img, bit_depth=16)
            target_img = tensor_to_img(target_img, bit_depth=16)

            tile_path = os.path.join(inference_args.output_folder, tile_name)
            os.makedirs(tile_path, exist_ok=True)

            cond_img.save(os.path.join(tile_path, 'input.png'))
            output_img.save(os.path.join(tile_path, output_file_name))
            target_img.save(os.path.join(tile_path, 'target.png'))

            with open(os.path.join(tile_path, 'metadata.json'), 'w') as fp:
                json.dump(metadata, fp)


if __name__ == '__main__':
    main()
