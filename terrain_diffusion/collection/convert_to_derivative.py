
import os

from tqdm import tqdm

from .mask import MaskArguments, generate_mask
from ..core.utils import (
    list_all_files,
    get_tiles_list,
    load_numpy,
    save_image,
)
from ..core.dir_args import CoverageArguments, DerivativeArguments, ElevationArguments
from ..core.tiles import generate_tiles, get_invalid_list, PER_PIXEL_TILE_ZOOM
from ..core.dataclass_argparser import CustomArgumentParser
from ..core.constants import (
    LODS,
    GRAD_DOF,
    USA_FROM,
    USA_TO,
    NP_EXT,
    PNG_EXT
)
from ..core.derivative import (
    elevation_to_SGF,
    SGF_to_image,
)


def create_derivative_data(tiles, elev_args: ElevationArguments, deriv_args: DerivativeArguments):

    valid_elevation_output = os.path.join(
        elev_args.elevation_dir, elev_args.valid_dir)
    valid_elevation_files = get_tiles_list(
        valid_elevation_output, NP_EXT, recursive=False)
    invalid_elevation_files = get_invalid_list(elev_args.elevation_dir)

    processed_deriv_files = set(list_all_files(deriv_args.derivative_dir))

    yres = xres = LODS[tiles['zoom']][0]

    with tqdm(total=tiles['total']) as progress:
        valid_count = 0
        for zoom, t_y, t_x, valid in tiles['tiles']:
            progress.update()
            if not valid:  # Not in mask
                continue

            tile = f'{zoom}_{t_y}_{t_x}'

            deriv_tile_name = f'{tile}.{PNG_EXT}'
            if deriv_tile_name in processed_deriv_files:
                continue  # already processed

            if tile not in valid_elevation_files:
                continue  # elevation file DNE
            if tile in invalid_elevation_files:
                continue  # elevation file is invalid

            data = load_numpy(os.path.join(
                valid_elevation_output, f'{tile}.{NP_EXT}'))

            gradient_field = elevation_to_SGF(data, xres, yres)

            valid_count += 1
            progress.set_description(f'Processing {tile} ({valid_count})')

            # Saving as image:
            deriv_tile_output = os.path.join(
                deriv_args.derivative_dir, deriv_args.raw_dir, deriv_tile_name)
            img = SGF_to_image(gradient_field)
            save_image(deriv_tile_output, img)

def main():

    parser = CustomArgumentParser(
        (ElevationArguments, DerivativeArguments, MaskArguments, CoverageArguments),
        description='Generate and filter derivative images'
    )

    elev_args, deriv_args, mask_args, cov_args = parser.parse_args_into_dataclasses()

    mask = generate_mask(mask_args, cov_args)

    print('Create derivative data')
    deriv_tiles = generate_tiles(
        USA_FROM, USA_TO,
        elev_args.elevation_zoom,
        mask,
        PER_PIXEL_TILE_ZOOM
    )
    create_derivative_data(
        deriv_tiles,
        elev_args,
        deriv_args
    )


if __name__ == '__main__':
    main()
