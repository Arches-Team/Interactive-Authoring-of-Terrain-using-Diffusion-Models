import numpy as np
import bpy
from bpy_extras.io_utils import ImportHelper

from .. import utils
from ...terrain_diffusion.labelling.label import generate_conditioning

# TODO: use a prop instead of import


class ExtractionProperty(bpy.types.PropertyGroup):
    range: bpy.props.FloatProperty(default=500, min=10, max=8000, description='Terrain range in meters')
    detailed_percentage: bpy.props.FloatProperty(default=0.5, min=0., max=1., description='How much the sketch is detailed in percentage. 1 as much as possible, 0 no sketch at all')



class ExtractSketch(bpy.types.Operator, ImportHelper):
    bl_idname = 'terraindm.extract_sketch'
    bl_label = 'Extract sketch'
    filter_glob: bpy.props.StringProperty(
        default='*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp', options={'HIDDEN'})

    def execute(self, context):
        extraction_props = context.scene.extraction_props

        # TODO: use prop
        bpy_img = bpy.data.images.load(self.filepath, check_existing=True)
        img = utils.to_np_array(bpy_img, grayscale=True).astype(np.float32)

        terrain_range = extraction_props.range
        detailed_percentage = 1 - extraction_props.detailed_percentage

        dem = img * terrain_range

        # TODO add resolution option
        sketch = generate_conditioning(dem, 19.11, base_threshold_range=(
            detailed_percentage, detailed_percentage)).astype(np.float32) / 255.

        sketch = utils.rgba_to_rgb(sketch)

        name = context.scene.generation_props.input_sketch
        utils.to_bpy_img(sketch, name)
        utils.update_2d_3d_views_img(name)

        return {'FINISHED'}
