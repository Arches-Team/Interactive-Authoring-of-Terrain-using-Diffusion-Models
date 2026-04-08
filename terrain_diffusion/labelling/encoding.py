import torch
import numpy as np

from ..core.utils import normalise_array
from ..core.derivative import elevation_to_SGF
from ..core.shared import ArgsKwargsWrapper

# Bins for absolute value of SGFs
SGF_BINS = [
    0.0,
    0.0024977277498692274, 0.00668445834890008,
    0.01155373640358448, 0.01762024126946926,
    0.024745840579271317, 0.032935578376054764,
    0.04226625710725784, 0.053384799510240555,
    0.06668040156364441, 0.08297241479158401,
    0.10330004245042801, 0.1289672553539276,
    0.1617264449596405, 0.2041773498058319,
    0.2624289095401764, 1.0
]

ELEV_BINS = [
    0.0,
    0.07188525050878525, 0.13861295580863953,
    0.19481192529201508, 0.24499885737895966,
    0.29208821058273315, 0.33713284134864807,
    0.3810635507106781, 0.42468908429145813,
    0.46863508224487305, 0.5136644244194031,
    0.5607843399047852, 0.6108644008636475,
    0.6658884286880493, 0.729381263256073,
    0.8109865188598633, 1.0
]

ELEV_RANGE_BINS = [
    # (11.17, 2122.78)
    0, 76.97, 98.8125,
    117.0, 139.22, 165.43,
    212.76, 310.66625, 398.9,
    466.62, 519.06, 574.575,
    635.68, 700.1825, 781.52,
    908.6875, 10000
]


def compute_bins(data, num_bins):
    # Compute bins so that data follows a uniform distribution
    step_size = 100/num_bins
    percentiles = np.arange(0, 100+step_size, step_size)
    bins = np.percentile(data, percentiles)
    return bins


def binify(data, bins):
    # Count number of elements that fall within each bin
    # Return normalised
    height = np.histogram(data, bins)[0]
    total = np.sum(height)
    return height / total


def uniform(bins):
    num_items = len(bins) - 1
    return np.ones(num_items) / num_items


class TerrainEncoder:

    @property
    def cross_attention_dim(self):
        raise NotImplementedError

    def baseline(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class TerrainStyle(ArgsKwargsWrapper):

    def __init__(self, terrains, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.terrains = terrains

    def to(self, device):
        self.terrains = self.terrains.to(device)
        return self


class GlobalTerrainStyle(TerrainStyle):

    def __init__(self,
                 terrains,
                 ranges,
                 resolutions,
                 *args, **kwargs) -> None:
        super().__init__(terrains, *args, **kwargs)

        self.ranges = ranges
        self.resolutions = resolutions


class SatelliteTerrainStyle(TerrainStyle):
    pass


def generate_features(values, bins):
    features = binify(values, bins)
    offset = uniform(bins)
    return features - offset


class GlobalTerrainEncoder(TerrainEncoder):

    @property
    def cross_attention_dim(self):
        return 48

    def baseline(self, batch_size):
        # Encoding of "nothing". Used for classifier free guidance
        return torch.zeros(batch_size, 1, self.cross_attention_dim)

    def _encode(self, terrain, range, resolution):
        """
        Given a piece of terrain, construct a low dimensional embedding that
        conveys global structure and type.

        Used for:
        - conditioning/encoder_hidden_states for UNet
        - clustering

        (batch, sequence_length, feature_dim)
        - sequence_length: terrain types a user can select (1+)
        - feature_dim: 1D vector where each component encodes some aspect of the terrain
        - [16] unnormalised elevation range distribution/histogram
        - [16] normalised elevation range distribution/histogram
        - [16] transformed slope-angle distribution/histogram
        """

        if terrain.ndim == 3:
            terrain = terrain[0]  # (1, 256, 256) -> (256, 256)

        terrain = range * (terrain + 1)/2  # Scale to (0, max)
        sgf = elevation_to_SGF(terrain, xres=resolution, yres=resolution)

        # Flatten and take absolute value (since terrains can be flipped arbitrarily)
        #  - flattening treats dh_dx and dh_dy the same
        #  - absolute values ignore direction
        sgf = np.abs(sgf.flatten())

        return self._generate_features(terrain, sgf)

    def _generate_features(self, terrain, sgf):
        features_1 = generate_features(terrain, ELEV_RANGE_BINS)
        features_2 = generate_features(normalise_array(terrain), ELEV_BINS)
        features_3 = generate_features(sgf, SGF_BINS)

        # Concatenate to form a single feature vector
        return np.concatenate((features_1, features_2, features_3)).astype(np.float32)

    def _prepare_data(self, terrain_style):
        to_return = []

        for i in (terrain_style.terrains, terrain_style.ranges, terrain_style.resolutions):
            if isinstance(i, torch.Tensor):
                i = i.cpu().numpy()
            to_return.append(i)

        return to_return

    def __call__(self, terrain_style: GlobalTerrainStyle):
        # NOTE: all variables are batched
        terrains, ranges, resolutions = self._prepare_data(terrain_style)
        batched_encodings = []
        for terrain, range, resolution in zip(terrains, ranges, resolutions):
            enc = self._encode(terrain, range, resolution)
            batched_encodings.append(torch.from_numpy(enc).unsqueeze(0))

        batched_encodings = torch.stack(batched_encodings)
        return batched_encodings


SATELLITE_HIST_R = [
    0.0, 0.12156862745098039, 0.15294117647058825,
    0.17647058823529413, 0.19607843137254902, 0.21568627450980393,
    0.23529411764705882, 0.25882352941176473, 0.2823529411764706,
    0.3137254901960784, 0.35294117647058826, 0.403921568627451,
    0.4666666666666667, 0.5294117647058824, 0.592156862745098,
    0.6588235294117647, 1.0
]

SATELLITE_HIST_G = [
    0.0, 0.19215686274509805, 0.22745098039215686,
    0.24705882352941178, 0.26666666666666666, 0.2823529411764706,
    0.2980392156862745, 0.3137254901960784, 0.3333333333333333,
    0.35294117647058826, 0.3803921568627451, 0.4117647058823529,
    0.4549019607843137, 0.5058823529411764, 0.5529411764705883,
    0.611764705882353, 1.0
]

SATELLITE_HIST_B = [
    0.0, 0.10588235294117647, 0.12549019607843137,
    0.1411764705882353, 0.1568627450980392, 0.17254901960784313,
    0.19215686274509805, 0.21176470588235294, 0.23137254901960785,
    0.2549019607843137, 0.2784313725490196, 0.3137254901960784,
    0.35294117647058826, 0.396078431372549, 0.44313725490196076,
    0.49411764705882355, 1.0
]


class SatelliteTerrainEncoder(TerrainEncoder):

    @property
    def cross_attention_dim(self):
        return 48

    def baseline(self, batch_size):
        # Encoding of "nothing". Used for classifier free guidance
        return torch.zeros(batch_size, 1, self.cross_attention_dim)

    def _encode(self, terrain):
        if isinstance(terrain, torch.Tensor):
            terrain = terrain.cpu().numpy()

        # Convert (-1, 1) to (0, 1)
        terrain = (terrain + 1)/2

        features_1 = generate_features(terrain[0].flatten(), SATELLITE_HIST_R)
        features_2 = generate_features(terrain[1].flatten(), SATELLITE_HIST_G)
        features_3 = generate_features(terrain[2].flatten(), SATELLITE_HIST_B)

        # Concatenate to form a single feature vector
        return np.concatenate((features_1, features_2, features_3)).astype(np.float32)

    def __call__(self, terrain_style: SatelliteTerrainStyle):
        # NOTE: all variables are batched
        batched_encodings = []
        for terrain in terrain_style.terrains:
            enc = self._encode(terrain)
            batched_encodings.append(torch.from_numpy(enc).unsqueeze(0))

        batched_encodings = torch.stack(batched_encodings)
        return batched_encodings
