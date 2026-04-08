
import lpips
from pytorch_msssim import ssim, ms_ssim
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance, calculate_activation_statistics
from torch.nn.functional import adaptive_avg_pool2d
import json
from dataclasses import dataclass, field
import os

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


from .model import TerrainDiffusionPipeline
from ...core.dataclass_argparser import CustomArgumentParser
from ...core.terrain_dataset import TerrainDataset
from ...core.terrain_transforms import UnnormaliseTransform
from ...core.utils import array_to_image
from ...evaluation.metrics import mse
from ...core.utils import load_image


@dataclass
class EvaluationArguments:
    eval_folder: str = field(
        default='./data/evaluation/',
        metadata={
            'help': 'Path to input folder'
        }
    )
    output_file_name: str = field(
        default='output_0_250_1.0.png',
        metadata={
            'help': 'Output file name'
        }
    )
    target_file_name: str = field(
        default='target.png',
        metadata={
            'help': 'Target file name'
        }
    )


def calculate_fid_given_tensors(tensors_1, tensors_2, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics_of_tensors(tensors_1, model, batch_size,
                                                        dims, device, num_workers)
    m2, s2 = calculate_activation_statistics_of_tensors(tensors_2, model, batch_size,
                                                        dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


class ElevationDataset(torch.utils.data.Dataset):
    def __init__(self, items, transforms=None):
        self.items = items
        self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def get_activations_of_tensors(
        tensors,
        model,
        batch_size=50,
        dims=2048,
        device='cpu',
        num_workers=1
):

    model.eval()

    if batch_size > len(tensors):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(tensors)

    dataset = ElevationDataset(tensors)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(tensors), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics_of_tensors(
    tensors,
    model,
    batch_size=50,
    dims=2048,
    device='cpu',
    num_workers=1
):
    act = get_activations_of_tensors(
        tensors, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def main():
    parser = CustomArgumentParser(
        (
            EvaluationArguments,
        ),
        description='Run evaluation on a dataset'
    )

    eval_args, = parser.parse_args_into_dataclasses()

    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    loss_fn_alex = lpips.LPIPS(net='alex')

    # Set metrics
    running_mse = 0
    running_ssim = 0
    running_ms_ssim = 0
    running_lpips = 0

    output_tensors = []
    target_tensors = []

    tiles = os.listdir(eval_args.eval_folder)

    for tile in tiles:
        output_img_path = os.path.join(
            eval_args.eval_folder, tile, eval_args.output_file_name)
        target_img_path = os.path.join(
            eval_args.eval_folder, tile, eval_args.target_file_name)

        output_img = load_image(output_img_path)
        target_img = load_image(target_img_path)

        output = to_tensor(output_img)
        target = to_tensor(target_img)

        output_tensors.append(output.expand(3, -1, -1))
        target_tensors.append(target.expand(3, -1, -1))

        output = output.unsqueeze(0)
        target = target.unsqueeze(0)

        running_mse += mse(output, target).item()
        running_ssim += ssim(output, target, data_range=1).item()
        running_ms_ssim += ms_ssim(output, target, data_range=1).item()

        running_lpips += loss_fn_alex(output, target).item()

    count = len(tiles)

    print('MSE:', running_mse/count)
    print('SSIM:', running_ssim/count)
    print('MS-SSIM:', running_ms_ssim/count)
    print('LPIPS:', running_lpips/count)

    fid = calculate_fid_given_tensors(
        output_tensors,
        target_tensors,
        batch_size=2,
        device=device,
        dims=2048,
        num_workers=1
    )

    print('FID:', fid)


if __name__ == '__main__':
    main()
