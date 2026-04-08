import zlib
import base64
import threading
import datetime
import os
import random

import numpy as np
import bpy
from bpy.types import bpy_prop_collection

from . import settings


def to_np_array(bpy_img, grayscale=True):
    if bpy_img.filepath != '' and bpy_img.colorspace_settings.name == 'sRGB' and grayscale:
        bpy_img.colorspace_settings.name = 'Non-Color'
        bpy_img.reload()
    img = np.asarray(bpy_img.pixels)
    s = bpy_img.size

    # TODO: raise a proper error
    if s[0] == 0 or s[1] == 0:
        raise Exception

    img = np.resize(img, (s[0], s[1], bpy_img.channels))

    if grayscale:
        # TODO : maybe add warning or do the mean of channels
        img = img[:, :, 0]
    else:
        # TODO: keep alpha channel
        img = img[:, :, 0:3]

    return img


def to_bpy_img(img: np.ndarray, name: str):
    img = img.astype(np.float32)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    if img.shape[2] == 1:
        img = np.repeat(img, 4, axis=2)
        img[:, :, 3] = 1.
    elif img.shape[2] == 3:
        img = np.concatenate(
            (img, np.ones((*img.shape[0:2], 1), dtype=img.dtype)), axis=2)

    height, width = img.shape[1], img.shape[0]
    if name not in bpy.data.images:
        blender_img = bpy.data.images.new(
            name=name, width=width, height=height, alpha=False, is_data=False)
    else:
        blender_img = bpy.data.images[name]
        blender_img.scale(height, width)

    blender_img.pixels.foreach_set(img.ravel())

    return blender_img


# To show the flat area (alpha channel for the inference), different color of RGB was used to simulate the alpha.
# 'Full' color, for rivers rigdes and cliffs are at 0.99, and flat area at 0.75
# The brush is set to 'add' blending mode. For example, one pixel at 1. belongs to its class and to a flat area.
# To be counted as a flat area, all channels of this pixels must be at at least 0.75.
# One limitation of this method is to be counted as flat if all others classes (rivers, rigdes, cliffs) are paint on the same pixel
# which might leads to strange results given the inconsistency of the features.
def rgb_to_rgba(rgb: np.ndarray):
    assert rgb.shape[2] == 3

    rgba = np.zeros((*rgb.shape[0:2], 4), dtype=rgb.dtype)
    for i in range(3):
        rgba[:, :, i][rgb[:, :, i] >= 0.98] = 1.

    all_equal = np.logical_and(np.abs(
        rgb[:, :, 1] - rgb[:, :, 2]) < 0.01, np.abs(rgb[:, :, 0] - rgb[:, :, 2]) < 0.01)
    rgba[:, :][all_equal] = 0.

    rgba[:, :, 3] = 1.
    rgba[:, :, 3][np.logical_and(rgb[:, :, 0] >= 0.74, np.logical_and(
        rgb[:, :, 1] >= 0.74, rgb[:, :, 2] >= 0.74))] = 0.
    rgba = np.clip(rgba, 0., 1.)

    return rgba


def rgba_to_rgb(rgba: np.ndarray):
    assert rgba.shape[2] == 4

    rgba[rgba > 0] = 1.

    rgb = np.zeros((*rgba.shape[0:2], 3), dtype=rgba.dtype)
    for i in range(3):
        rgb[:, :, i][rgba[:, :, i] >= .999] = 0.99
        # Flat area
        rgb[:, :, i][rgba[:, :, 3] <= 0.001] += 0.75

    return rgb


def base64zlib_to_nparray(base64_input: str, shape, dtype=np.float32):
    if isinstance(base64_input, str):
        base64_input = base64_input.encode('utf-8')
    else:
        base64_input = base64_input.decode('utf-8')
    img_zlib = base64.b64decode(base64_input)
    img = np.frombuffer(zlib.decompress(img_zlib), dtype=dtype).reshape(shape)
    return img


def nparray_to_base64zlib(img: np.ndarray):
    data = zlib.compress(np.ascontiguousarray(img))
    im_b64 = base64.b64encode(data).decode('utf-8')
    return im_b64


def search(ID):
    def users(col):
        ret = tuple(repr(o) for o in col if o.user_of_id(ID))
        return ret if ret else None
    return filter(None, (
        users(getattr(bpy.data, p))
        for p in dir(bpy.data)
        if isinstance(
            getattr(bpy.data, p, None),
            bpy_prop_collection
        )
    )
    )


def update_2d_3d_views_img(name: str):
    if name not in bpy.data.images:
        # TODO: raise or alert user
        return

    blender_img = bpy.data.images[name]
    for user_str in search(blender_img):
        user = eval(user_str[0])
        user.update_tag()
    blender_img.update()


def deepcopy_object(obj, copy_mat: bool = True, show_in_3dview: bool = True, collection_name='tmp'):
    copy = bpy.context.active_object.copy()
    copy.data = copy.data.copy()
    if copy.animation_data:
        copy.animation_data.action = copy.animation_data.action.copy()

    if copy_mat:
        copy.active_material = copy.active_material.copy()

    if show_in_3dview:
        if collection_name not in bpy.data.collections:
            collection = bpy.data.collections.new(collection_name)
            bpy.context.collection.children.link(collection)

        bpy.data.collections[collection_name].objects.link(copy)

    return copy


def delete_object(obj, mat=None):
    if mat is not None:
        bpy.data.materials.remove(mat)
    bpy.data.objects.remove(obj, do_unlink=True)


def delete_collection(name):
    if name not in bpy.data.collections:
        return

    collection = bpy.data.collections[name]
    for obj in collection.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.data.collections.remove(collection)


# https://github.com/benrugg/AI-Render/blob/e81ad4976d339ba585c379ac85152ed914d483e1/utils.py#L50
def get_filepath_in_package(path, filename="", starting_dir=__file__):
    """Convert a relative path in the add-on package to an absolute path"""
    script_path = os.path.dirname(os.path.realpath(starting_dir))
    subpath = path + os.sep + filename if path else filename
    return os.path.join(script_path, subpath)


def random_max_int():
    # 2147483647 -> Blender max size (signed integer)
    return random.randint(0, 2147483647)

# https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self):
        super().__init__()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()


# https://deepnote.com/@rmi-ppin/Faie-un-singleton-en-python-0d187a73-2f24-4c49-b2aa-bbbf4a45ace6
class Singleton:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Singleton, cls).__new__(
                cls, *args, **kwargs)
        return cls.__instance


class Logger(Singleton):
    filename: str = 'log.txt'
    counter: int = 0
    init: bool = False

    def start(self, folder):
        now = datetime.datetime.now()
        self.main_folder = os.path.join(
            folder, f'{now.day:02d}-{now.month:02d}-{now.year}_{now.hour:02d}-{now.minute:02d}-{now.second:02d}')
        os.makedirs(self.main_folder, exist_ok=True)
        self.init = True
        Logger().log('Log session started.')

    def is_init(self):
        return self.init

    def log_image(self, img: np.ndarray, prefix: str = ''):
        if self.init:
            now = datetime.datetime.now()
            np.save(os.path.join(self.main_folder,
                                 f'{now.day:02d}_{now.month:02d}_{now.year}_{now.hour:02d}_{now.minute:02d}_{now.second:02d}{f"_{prefix}" if prefix else ""}_log_{self.counter}.npy'), img)
            self.counter += 1

    def log(self, txt: str):
        if self.init:
            now = datetime.datetime.now()
            with open(os.path.join(self.main_folder, self.filename), "a") as f:
                f.write(
                    f'{now.day:02d}/{now.month:02d}/{now.year} {now.hour:02d}:{now.minute:02d}:{now.second:02d} - {txt}\n')
