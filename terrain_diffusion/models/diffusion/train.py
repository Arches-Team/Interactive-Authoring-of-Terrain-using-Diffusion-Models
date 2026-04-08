
import os
import warnings
from typing import List, Optional
from dataclasses import dataclass, field, fields

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast
from torchvision.utils import make_grid, save_image

from diffusers.models import (
    UNet2DConditionModel,
    UNet2DModel
)
from diffusers.schedulers import DDIMScheduler
from tqdm.auto import trange

from .model import TerrainDiffusionPipeline, exists
from ...core.utils import (
    exists,
    tile_images,
    format_memory
)
from ...core.terrain_dataset import TerrainDataset
from ...core.terrain_transforms import (
    NormaliseTransform,
    UnnormaliseTransform,
)
from ...core.dataclass_argparser import CustomArgumentParser
from ...training.trainer import (
    BaseTrainer,
    TrainingArguments,
    ModelInputs,
    run_if,
)
from ...labelling.encoding import (
    GlobalTerrainEncoder,
    SatelliteTerrainEncoder,
)

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class SamplingArguments:
    sample_steps: int = field(
        default=1000,
        metadata={
            'help': 'Run sampling every X steps'
        }
    )
    num_samples: int = field(
        default=8,
        metadata={
            'help': 'Number of samples to generate (must be divisible by batch_size)'
        }
    )
    samples_dir: str = field(
        default='samples',
        metadata={
            'help': 'Where to save sample images'
        }
    )
    normalise_output: bool = field(
        default=False,
        metadata={
            'help': 'Whether to normalise output images after sampling'
        }
    )


@dataclass
class ImageArguments:
    target_image_channels: int = field(
        default=1,
        metadata={
            'help': 'Number of channels in the target image (in pixel space)'
        }
    )
    cond_image_channels: int = field(
        default=1,  # TODO
        metadata={
            'help': 'Number of channels in the conditional image'
        }
    )

    target_image_type: str = field(
        default='elevation',
        metadata={
            'choices': ['elevation', 'satellite'],
            'help': 'Type of image to generate',
        }
    )


@dataclass
class DiffusionArguments:
    # Default params based on:
    # https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/unet/config.json

    unet_block_out_channels: List[int] = field(
        default_factory=lambda: [
            64, 128, 256, 512
        ],
        metadata={
            'help': 'List of block output channels',
            'nargs': '+'
        }
    )
    unet_down_block_types: List[str] = field(
        default_factory=lambda: {
            'none': [
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ],
            'attn': [
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ],
            'crossattn': [
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ]
        },
        metadata={
            'help': 'List of downsample block types.',
            'nargs': '+'
        }
    )
    unet_up_block_types: List[str] = field(
        default_factory=lambda: {
            'none': [
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ],
            'attn': [
                "UpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
            ],
            'crossattn': [
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ]
        },
        metadata={
            'help': 'List of upsample block types.',
            'nargs': '+'
        }
    )
    unet_layers_per_block: int = field(
        default=3,
        metadata={
            'help': 'The number of layers per block.'
        }
    )
    unet_attention_head_dim: int = field(
        default=8,
        metadata={
            'help': 'The attention head dimension.'
        }
    )

    unet_use_embeddings: bool = field(
        default=False,
        metadata={
            'help': 'Whether to further condition denoising on embedding.'
        }
    )

    unet_attn_type: str = field(
        default='none',
        metadata={
            'choices': ['none', 'attn', 'crossattn'],
            'help': 'Type of attention to use',
        }
    )

    def __post_init__(self):
        for f in fields(self):
            k = getattr(self, f.name)
            if isinstance(k, dict):
                setattr(self, f.name, k[self.unet_attn_type])


@dataclass
class SchedulerArguments:
    # TODO add support for other scheduler params

    num_train_timesteps: int = field(
        default=1000,
        metadata={
            'help': 'Number of diffusion steps used to train the model.'
        }
    )
    num_inference_steps: int = field(
        default=50,
        metadata={
            'help': 'Number of diffusion steps used when generating samples.'
        }
    )
    eta: float = field(
        default=1.0,
        metadata={
            'help': 'Corresponds to parameter eta (η) in the DDIM paper. Only applies to'
                    'DDIMScheduler, will be ignored for others.'
        }
    )


@dataclass
class DiffusionTrainingArguments(TrainingArguments):
    results_dir: str = field(
        default='./models/diffusion',
        metadata=TrainingArguments.__dataclass_fields__['results_dir'].metadata
    )


@dataclass
class DatasetArguments:
    dataset_folder: str = field(
        default='./data/processed/',
        metadata={
            'help': 'Folder containing train, valid, and test subfolder'
        }
    )
    target_image: Optional[str] = field(
        default='elevation-256x256.png',
        metadata={
            'help': 'Path to target image inside tile folder'
        }
    )
    cond_image: Optional[str] = field(
        default=None,  # 'elevation-64x64.png'
        metadata={
            'help': 'Path to conditional image inside tile folder. If None, train unconditionally'
        }
    )


class DiffusionModelTrainer(BaseTrainer):
    def __init__(self,
                 model: TerrainDiffusionPipeline,
                 generator: Optional[torch.Generator] = None,
                 scheduler_args: Optional[SchedulerArguments] = None,
                 sampling_args: Optional[SamplingArguments] = None,
                 *args, **kwargs):

        # We set the U-Net to be the model to train
        super().__init__(model=model.unet, *args, **kwargs)

        if sampling_args is None:
            sampling_args = SamplingArguments()

        assert sampling_args.num_samples % self.training_args.batch_size == 0, 'Number of samples must be divisible by batch_size'
        self.sampling_args = sampling_args

        self.unnormalise = UnnormaliseTransform()
        self.normalise = NormaliseTransform()

        self.pipeline = model

        # Easier accessing
        self.terrain_encoder = model.terrain_encoder
        self.noise_scheduler = model.scheduler

        self.generator = generator

        if scheduler_args is None:
            scheduler_args = SchedulerArguments()
        self.scheduler_args = scheduler_args

        # Needed since UNet2DModel returns UNet2DOutput object
        self.transform_outputs = lambda x: x.sample

        self.latents_dtype = next(self.model.parameters()).dtype

    def train(self):
        # Run normal training loop
        super().train()

        # Save diffusers pipeline after training
        final_dir = os.path.join(self.training_args.results_dir, 'final')
        self.pipeline.save_pretrained(final_dir)

    def extract(self, item):

        # 1. Extract images
        target_image = item['target_image'].to(self.device)
        batch_size = target_image.shape[0]

        # 2. Sample noise that we'll add to the target_image
        noise = torch.randn(
            target_image.shape,
            generator=self.generator,
            dtype=target_image.dtype,
            device=target_image.device
        )

        # 3. Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
            generator=self.generator,
        ).long()

        # 4. Add noise to the target latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_target_image = self.noise_scheduler.add_noise(
            target_image, noise, timesteps)

        inputs = [noisy_target_image]
        if exists(item.get('cond_image')):
            cond_image = item['cond_image'].to(self.device)
            if cond_image.shape[-2:] != noisy_target_image.shape[-2:]:
                # Resize conditional image if needed
                cond_image = F.interpolate(
                    cond_image,
                    size=noisy_target_image.shape[-2:]
                )
            inputs.append(cond_image)

        # 5. Concat target and conditional latents in the channel dimension.
        sample = torch.cat(inputs, dim=1)

        # 6. Prepare style, if needed
        other_model_inputs = {}
        terrain_style = self.pipeline.create_terrain_style(
            target_image, item['metadata'])
        if exists(terrain_style):
            other_model_inputs['encoder_hidden_states'] = self.pipeline._encode_terrain_style(
                terrain_style,
                do_classifier_free_guidance=False
            )

        inputs = ModelInputs(
            sample=sample,
            timestep=timesteps,
            **other_model_inputs
        )
        return inputs, noise

    @run_if(lambda self: self.steps % self.sampling_args.sample_steps == 0 or self.steps == self.total_train_steps)
    def sample(self):
        self.model.eval()

        dl = iter(self.dataloaders['valid'])

        num_samples = min(self.sampling_args.num_samples, len(dl))
        num_samples -= num_samples % self.training_args.batch_size  # Subtract offset

        num_iters = num_samples // self.training_args.batch_size

        # Create generator for reproducibility
        generator = None
        if exists(self.training_args.seed):
            generator = torch.Generator(self.model.device)
            generator.manual_seed(self.training_args.seed)

        images = []
        sampling_info = f'Sampling | epoch={self.epoch:.2f}, steps={self.steps}'

        with trange(num_iters, desc=sampling_info) as progress, torch.no_grad(), autocast(enabled=self.training_args.amp):
            for _ in progress:
                item = next(dl)
                target_image = item['target_image'].to(self.device)

                to_zip = []
                cond_image = None
                pipeline_inputs = {}
                # (b) encode conditional image into its latent space
                if exists(item.get('cond_image')):
                    cond_image = item['cond_image'].to(self.device)

                    resized_inputs = cond_image
                    if resized_inputs.shape[-2:] != target_image.shape[-2:]:
                        # Resize conditional image if needed
                        resized_inputs = F.interpolate(
                            resized_inputs, size=target_image.shape[-2:])

                    to_zip.append(resized_inputs)

                    pipeline_inputs['cond_image'] = resized_inputs

                else:
                    pipeline_inputs['output_size'] = target_image.shape

                terrain_style = self.pipeline.create_terrain_style(
                    target_image, item['metadata'])

                # Pass inputs through the pipeline
                outputs = self.pipeline(
                    num_inference_steps=self.scheduler_args.num_inference_steps,
                    eta=self.scheduler_args.eta,
                    generator=generator,
                    normalise_output=True,
                    terrain_style=terrain_style,
                    **pipeline_inputs,
                ).images

                to_zip.append(outputs)
                if exists(cond_image) or exists(self.terrain_encoder):
                    # Display target image if conditional image or encoding provided
                    to_zip.append(target_image)

                to_zip = tile_images(to_zip)
                for row in zip(*to_zip):
                    images.extend(row)

        nrow = len(images) // num_samples

        grid = self.unnormalise(make_grid(images, nrow=nrow))

        out_dir = os.path.join(
            self.training_args.results_dir, self.sampling_args.samples_dir)
        os.makedirs(out_dir, exist_ok=True)
        save_image(grid, os.path.join(out_dir, f'sample-{self.steps}.png'))


def main():
    parser = CustomArgumentParser(
        (
            DatasetArguments,

            ImageArguments,

            DiffusionTrainingArguments, SamplingArguments,

            # Diffusion arguments
            DiffusionArguments, SchedulerArguments
        ),
        description='Train a diffusion model',
        allow_abbrev=False,
    )

    (
        dataset_args,
        image_args,
        training_args, sampling_args,
        diffusion_args, scheduler_args
    ) = parser.parse_args_into_dataclasses()


    # Parse arguments and display using logger
    logger.info(f'{" Training with arguments ":*^40}')
    logger.info(dataset_args)
    logger.info(image_args)
    logger.info(training_args)
    logger.info(sampling_args)
    logger.info(diffusion_args)
    logger.info(scheduler_args)
    logger.info('-'*40 + '\n')

    # Get device info and display using logger
    logger.info(f'{" Device information ":*^40}')
    device_count = torch.cuda.device_count()
    logger.info(f'device_count={device_count}')
    if device_count > 1:
        logger.warning('Multiple devices detected, choosing the first.')

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    logger.info(f'CUDA_VISIBLE_DEVICES={visible_devices}')

    # setting device on GPU if available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    if device.type == 'cuda':  # additional info when using cuda
        device_properties = torch.cuda.get_device_properties(device)
        logger.info(f'Device name: {device_properties.name}')
        logger.info(
            f'Total memory: {format_memory(device_properties.total_memory)}')
        logger.info(
            f'Allocated memory: {format_memory(torch.cuda.memory_allocated(device))}')
        logger.info(
            f'Cached memory: {format_memory(torch.cuda.memory_reserved(device))}')

    logger.info('-'*40 + '\n')

    # In image space
    unet_out_channels = unet_in_channels = image_args.target_image_channels

    # Add additional channels if conditional generation
    if exists(dataset_args.cond_image):
        unet_in_channels += image_args.cond_image_channels

    model_kwargs = dict(
        in_channels=unet_in_channels,
        out_channels=unet_out_channels,
        block_out_channels=diffusion_args.unet_block_out_channels,
        down_block_types=diffusion_args.unet_down_block_types,
        up_block_types=diffusion_args.unet_up_block_types,
        layers_per_block=diffusion_args.unet_layers_per_block,
        attention_head_dim=diffusion_args.unet_attention_head_dim,
    )

    terrain_encoder = None
    if diffusion_args.unet_attn_type == 'crossattn' or diffusion_args.unet_use_embeddings:
        model_class = UNet2DConditionModel

        if diffusion_args.unet_use_embeddings:
            if image_args.target_image_type == 'elevation':
                terrain_encoder = GlobalTerrainEncoder()
            elif image_args.target_image_type == 'satellite':
                terrain_encoder = SatelliteTerrainEncoder()
            else:
                raise ValueError(
                    f'Invalid `--target_image_type` specified: {image_args.target_image_type}')

            unet_cross_attention_dim = terrain_encoder.cross_attention_dim
        else:
            unet_cross_attention_dim = 48

        model_kwargs.update(
            cross_attention_dim=unet_cross_attention_dim
        )
    else:
        model_class = UNet2DModel

    model = model_class(**model_kwargs)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=scheduler_args.num_train_timesteps,
        beta_start=1e-6,
        beta_end=1e-2,
        beta_schedule='linear',
        clip_sample=False,
    )
    noise_scheduler.set_timesteps(scheduler_args.num_inference_steps)

    pipeline = TerrainDiffusionPipeline(
        unet=model,
        terrain_encoder=terrain_encoder,
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(device)

    trainer_class = DiffusionModelTrainer
    trainer_kwargs = dict(
        model=pipeline,

        # Additional arguments
        training_args=training_args,
        scheduler_args=scheduler_args,
        sampling_args=sampling_args,
    )

    index_mapping = dict(
        cond_image=dataset_args.cond_image,
        target_image=dataset_args.target_image
    )

    # Move model to device
    model = model.to(device)

    shared_dataset_kwargs = dict(
        index_mapping=index_mapping,
        seed=training_args.seed,
        remove_incomplete_tiles=False,
    )
    datasets = dict(
        train=TerrainDataset(
            folder=os.path.join(dataset_args.dataset_folder, 'train'),
            **shared_dataset_kwargs
        ),
        valid=TerrainDataset(
            folder=os.path.join(dataset_args.dataset_folder, 'valid'),
            data_augmentation=False,
            **shared_dataset_kwargs
        ),
        test=TerrainDataset(
            folder=os.path.join(dataset_args.dataset_folder, 'test'),
            data_augmentation=False,
            **shared_dataset_kwargs
        )
    )

    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_args.lr,
        betas=(0.9, 0.999),
        weight_decay=1e-5,  # https://stackoverflow.com/a/46597531/13989043
    )

    # Define loss function
    criterion = nn.MSELoss()

    # Define scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=training_args.lr_scheduler_factor,
        patience=training_args.lr_scheduler_patience,
        threshold=training_args.lr_scheduler_threshold,
        min_lr=training_args.lr_scheduler_min_lr,
        verbose=training_args.lr_scheduler_verbose,
    )

    trainer = trainer_class(
        datasets=datasets,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        **trainer_kwargs
    )
    trainer.train()


if __name__ == '__main__':
    main()
