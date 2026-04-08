from dataclasses import dataclass
import inspect
from typing import Optional, Tuple, Union, List, Generator
import os

import torch
import torch.utils.checkpoint

from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models import (
    UNet2DModel,
    UNet2DConditionModel,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ...labelling.encoding import (
    TerrainEncoder,
    TerrainStyle,
    GlobalTerrainEncoder,
    GlobalTerrainStyle,
    SatelliteTerrainEncoder,
    SatelliteTerrainStyle
)
from ...core.utils import exactly_one_exists, exists


TERRAIN_ENCODER_FOLDER = 'terrain_encoder'
TERRAIN_ENCODER_NAME = 'terrain_encoder.bin'


def return_feedback_decorator(function):
    def wrapper(*args, **kwargs):
        kwargs['return_feedback'] = kwargs.pop('return_feedback', False)
        result = function(*args, **kwargs)
        if not kwargs['return_feedback']:
            result = next(result)
        return result
    return wrapper


@dataclass
class TerrainDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Terrain Diffusion pipelines.

    Args:
        images (`torch.Tensor`)
            Denoised images as torch.Tensor array of shape `(batch_size, height, width, num_channels)`
    """

    images: torch.Tensor
    step: int
    num_inference_steps: int


class TerrainDiffusionPipeline(DiffusionPipeline):
    r"""
    A pipeline for image-to-image translation

    This class inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DConditionModel`]): U-Net architecture to denoise the encoded image.
        terrain_encoder ([`TerrainEncoder`]):
            Frozen terrain-encoder. Used for stylisation of terrain.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`EulerDiscreteScheduler`],
            [`EulerAncestralDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        unet: Union[UNet2DModel, UNet2DConditionModel],
        terrain_encoder: Optional[TerrainEncoder],
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
        ],
    ):
        if isinstance(unet, UNet2DConditionModel) and terrain_encoder is None:
            raise ValueError(
                '`terrain_encoder` must be specified when using `UNet2DConditionModel`')

        super().__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            terrain_encoder=terrain_encoder
        )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = False,
    ):
        # Workaround to save terrain_encoder separately
        super().save_pretrained(save_directory, safe_serialization)

        if exists(self.terrain_encoder):
            terrain_encoder_folder = os.path.join(
                save_directory, TERRAIN_ENCODER_FOLDER)
            os.makedirs(terrain_encoder_folder, exist_ok=True)
            torch.save(self.terrain_encoder, os.path.join(
                terrain_encoder_folder, TERRAIN_ENCODER_NAME))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs) -> 'TerrainDiffusionPipeline':
        terrain_encoder_path = os.path.join(
            pretrained_model_name_or_path, TERRAIN_ENCODER_FOLDER, TERRAIN_ENCODER_NAME)

        if os.path.exists(terrain_encoder_path):
            kwargs[TERRAIN_ENCODER_FOLDER] = torch.load(terrain_encoder_path)

        return super().from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )

    def prepare_inputs(self,
                       batch_size, num_noise_channels, unet_input_height, unet_input_width,
                       dtype, device, generator,
                       ):

        shape = (batch_size, num_noise_channels,
                 unet_input_height, unet_input_width)

        inputs = randn_tensor(shape, generator=generator,
                              device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        inputs = inputs * self.scheduler.init_noise_sigma
        return inputs

    def create_terrain_style(self, target_image, metadata=None):
        if self.terrain_encoder is None:
            return None

        if isinstance(self.terrain_encoder, GlobalTerrainEncoder):
            terrain_style = GlobalTerrainStyle(
                terrains=target_image,
                ranges=metadata['range'],
                resolutions=metadata['resolution'],
            )
        elif isinstance(self.terrain_encoder, SatelliteTerrainEncoder):
            terrain_style = SatelliteTerrainStyle(terrains=target_image)
        else:
            terrain_style = TerrainStyle(terrains=target_image)

        return terrain_style

    def _encode_terrain_style(self,
                              terrain_style: TerrainStyle,
                              do_classifier_free_guidance=False
                              ):
        # TODO add negative_prompts

        # Generate encodings based on normalised target_image
        encoder_hidden_states = self.terrain_encoder(terrain_style)

        if do_classifier_free_guidance:
            baseline = self.terrain_encoder.baseline(
                batch_size=terrain_style.terrains.shape[0])

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            encoder_hidden_states = torch.cat(
                [baseline, encoder_hidden_states])

        encoder_hidden_states = encoder_hidden_states.to(self.device)
        return encoder_hidden_states

    @return_feedback_decorator
    @torch.no_grad()
    def __call__(
        self,

        # Typically a sketch or low-res image
        cond_image: Optional[torch.Tensor] = None,
        output_size: Optional[Tuple[int]] = None,

        # Further condition on exemplars (which will be encoded using `self.terrain_encoder`)
        terrain_style: Optional[TerrainStyle] = None,

        eta: Optional[float] = 1.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        seed: Optional[Union[int, List[int]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 1,
        normalise_output=True,
        return_feedback=False,

        progress_bar=True,
        **kwargs,
    ) -> Union[Generator[TerrainDiffusionPipelineOutput, None, None], TerrainDiffusionPipelineOutput]:
        r"""
        Args:
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 1):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the `terrain_style`,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
        Returns:
            `Union[Generator[TerrainDiffusionPipelineOutput], TerrainDiffusionPipelineOutput]`
        """
        # TODO move comments to docstrings
        # TODO remove code duplication with trainer forward method
        # TODO add negative styles

        # 1. Check inputs. Raise error if not correct
        if not exactly_one_exists(cond_image, output_size):
            raise ValueError(
                'Must specify exactly one of `cond_image` or `output_size`.'
            )

        if isinstance(self.unet, UNet2DConditionModel) and terrain_style is None:
            raise ValueError(
                '`terrain_style` must be specified when using `UNet2DConditionModel`')

        # 2. Define call parameters
        device = self.device
        dtype = next(self.unet.parameters()).dtype
        if exists(seed):
            if exists(generator):
                raise ValueError(
                    'May not specify `generator` and `seed` at the same time'
                )

            # Seed specified, but generator not specified.
            if isinstance(seed, list):
                # Create list of generators
                generator = [torch.Generator(
                    device).manual_seed(s) for s in seed]
            else:
                generator = torch.Generator(device).manual_seed(seed)

        # 3. Set unet parameters
        if exists(cond_image):
            if not isinstance(cond_image, torch.Tensor):
                raise ValueError(
                    f"If specified, `cond_image` has to be of type `torch.Tensor` but is {type(cond_image)}"
                )

            batch_size, _, height, width = cond_image.shape

            # Ensure RGBA image
            # if cond_channels != 4:
            #     raise ValueError(
            #         f'Invalid number of conditional channels. Expected 4 but got {cond_channels}.')

        else:  # Unconditional generation
            batch_size, height, width = output_size

            if len(output_size) != 3:
                raise ValueError(
                    '`output_size` must be of the form (num_samples, height, width)')

            # Construct "empty" conditional image
            cond_image = torch.cat(
                torch.full(
                    (batch_size, 3, height, width),
                    fill_value=-1,
                    device=device, dtype=dtype
                ),
                torch.full(
                    (batch_size, 1, height, width),
                    fill_value=1,
                    device=device, dtype=dtype
                ),
                dim=1  # Concatenate on channel
            )

        num_noise_channels = self.unet.config.in_channels - cond_image.shape[1]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 4. Encode `terrain_styles` if needed
        unet_kwargs = {}
        if exists(self.terrain_encoder):
            unet_kwargs['encoder_hidden_states'] = self._encode_terrain_style(
                terrain_style, do_classifier_free_guidance
            )

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare inputs (noise) for unet
        # (i.e., what we will be denoising)
        unet_inputs = self.prepare_inputs(
            batch_size,
            num_noise_channels,
            height,
            width,
            dtype,
            device,
            generator
        )

        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Add progress bar if wanted
        if progress_bar:
            timesteps = self.progress_bar(timesteps)

        # 9. Denoising loop
        output = None
        for i, t in enumerate(timesteps):
            # concat target and conditional latents in the channel dimension.
            # NOTE: will throw error if latent width and height are not the same
            # (number of channels may differ)
            inputs = [unet_inputs]
            if cond_image is not None:
                inputs.append(cond_image)
            model_inputs = torch.cat(inputs, dim=1)

            # expand the model_inputs if we are doing classifier free guidance
            model_inputs = torch.cat(
                [model_inputs] * 2) if do_classifier_free_guidance else model_inputs
            model_inputs = self.scheduler.scale_model_input(model_inputs, t)

            # predict the noise residual
            noise_pred = self.unet(
                sample=model_inputs,
                timestep=t,
                **unet_kwargs
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            stepped = self.scheduler.step(
                noise_pred, t, unet_inputs, **extra_step_kwargs)
            unet_inputs = stepped.prev_sample

            # Post-process original if user wants all outputs or it is the final iteration
            if return_feedback or t == self.scheduler.timesteps[-1]:
                pred_original_sample = self.postprocess(
                    stepped.pred_original_sample, normalise_output)

                output = TerrainDiffusionPipelineOutput(
                    images=pred_original_sample,
                    step=i,
                    num_inference_steps=num_inference_steps
                )

                # Yield current prediction of start if user wants
                if return_feedback:
                    yield output

        # Have final yield outside of the progress bar
        if exists(output) and not return_feedback:
            yield output

    def postprocess(self, outputs, normalise_output=True):
        if normalise_output:
            batched_views = outputs.view(outputs.shape[0], -1)
            min_vals, _ = torch.min(batched_views, dim=1)
            max_vals, _ = torch.max(batched_views, dim=1)

            image_ranges = (max_vals - min_vals)[:, None, None, None]

            outputs = (
                2 * (outputs - min_vals[:, None, None, None]) / image_ranges) - 1

        else:
            outputs = torch.clamp(outputs, -1.0, 1.0)

        return outputs

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        extra_step_kwargs = {}

        scheduler_params = set(inspect.signature(
            self.scheduler.step).parameters.keys())

        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = 'eta' in scheduler_params
        if accepts_eta:
            extra_step_kwargs['eta'] = eta

        # check if the scheduler accepts generator
        accepts_generator = 'generator' in scheduler_params
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs
