import math
import time

import bpy
from bpy.app.handlers import persistent
import numpy as np

from ..generation import sketch
from .. import utils


current_displayed_object = None


def update_displacement_range(self, context):
    global current_displayed_object

    value = context.scene.generation_props.terrain_range
    value = max(100, value)
    t = value / 1200

    # Displacement is too high after 0.5
    t = math.tanh(t * 2) / 2

    if current_displayed_object is not None:
        for modifier in current_displayed_object.modifiers:
            if modifier.type == "DISPLACE":
                modifier.strength = t

    for modifier in bpy.context.active_object.modifiers:
        if modifier.type == "DISPLACE":
            modifier.strength = t


def enable_display_generation_on_realtime(self, context):
    if context.scene.generation_props.update_real_time:
        context.scene.generation_props.display_generation = True


paint_copy = None
paint_mat = None


def update_display_generation(self, context):
    global paint_copy
    global paint_mat
    global current_displayed_object

    is_display = context.scene.generation_props.display_generation

    # If changing the terrain in real time, create a copy with the displacement modifier and hide the original (with displacement disabled)
    # Allow the user to draw on the mesh while generating
    if is_display and paint_copy is None:
        active_obj = context.active_object
        paint_copy = utils.deepcopy_object(
            active_obj, copy_mat=True, show_in_3dview=True, collection_name='generation_tmp')
        paint_mat = paint_copy.active_material
        active_obj.hide_set(True)

        for modifier in active_obj.modifiers:
            if modifier.type == "DISPLACE":
                modifier.show_viewport = False

        current_displayed_object = paint_copy
    elif not is_display and paint_copy is not None:
        obj = context.object
        obj.hide_set(False)
        for modifier in obj.modifiers:
            if modifier.type == "DISPLACE":
                modifier.show_viewport = True

        utils.delete_object(paint_copy, paint_mat)
        paint_copy = None
        paint_mat = None

        current_displayed_object = context.active_object


def update_image_size(self, context):
    generation_props = context.scene.generation_props
    size = int(generation_props.image_size)
    bpy.data.images[generation_props.input_sketch].scale(size, size)


# Needed by the 'set' function
def get_input_sketch(self):
    if not 'input_sketch' in self:
        return ''
    return self['input_sketch']


def set_input_sketch(self, value):
    old_value = value
    if 'input_sketch' in self:
        old_value = self['input_sketch']

    self['input_sketch'] = value

    if value != '':
        # TODO: not always active object
        active_obj = bpy.context.active_object
        for node in active_obj.active_material.node_tree.nodes:
            if node.bl_idname == 'ShaderNodeTexImage' and node.image.name == old_value:
                node.image = bpy.data.images[value]


def enum_sample_preview(self, context):
    enum_items = []

    cpt = 0
    for img in bpy.data.images:
        name = img.name
        if img.name.startswith('ld_sketch_sample'):
            enum_items.append((name, str(
                context.scene.generation_props.seed + cpt), "", img.preview_ensure().icon_id, cpt))
            cpt += 1

    return enum_items


# Needed by the 'set' function
def get_sample_preview(self):
    if not 'sample_preview' in self:
        return 0
    return self['sample_preview']


def set_sample_preview(self, value):
    old_value = value
    if 'sample_preview' in self:
        old_value = self['sample_preview']

    self['sample_preview'] = value

    for user_str in utils.search(bpy.data.images[old_value]):
        user = eval(user_str[0])
        user.image = bpy.data.images[value]
        user.update_tag()


class GenerationProperty(bpy.types.PropertyGroup):
    # TODO: connect seed
    seed: bpy.props.IntProperty(
        default=0, min=0, description='Seed of the generation')
    random_seed: bpy.props.BoolProperty(
        default=False, description='Choose a random seed at each inference')
    eta: bpy.props.FloatProperty(
        default=1., min=0., max=1., description='DDIM sampling hyperparameter')
    fp16: bpy.props.BoolProperty(
        default=True, description='Use 16-bit float precision for inference')

    nb_sample: bpy.props.IntProperty(
        default=1, min=1, max=20, description='Number of samples')
    sampling_steps: bpy.props.IntProperty(
        default=100, min=1, max=1000, description='Number of sampling steps (i.e. denoising steps)')
    guidance: bpy.props.FloatProperty(
        default=1, min=1, max=5, description='Classifier-free guidance scale')

    terrain_range: bpy.props.FloatProperty(
        default=500, min=10, max=1200, description='Terrain height dynamic in meters', update=update_displacement_range)
    terrain_resolution: bpy.props.FloatProperty(
        default=20, min=2, max=50, description='Terrain resolution in meters per pixel')

    # Real time or interactive
    update_real_time: bpy.props.BoolProperty(
        default=False, description='Real time rendering with every step', update=enable_display_generation_on_realtime)
    # update_each_stroke: bpy.props.BoolProperty(default=False, description='Real time rendering at each stroke',
    #                                            update=enable_real_time_with_update_at_strokes)
    display_generation: bpy.props.BoolProperty(
        default=False, description='Render intermediate results', update=update_display_generation)
    image_size: bpy.props.EnumProperty(items=[('128', '128x128', '', 0), ('256', '256x256', '', 1),
                                              ('512', '512x512', '', 2), ('1024', '1024x1024', '', 3)],
                                       description='Inference size', default='256', update=update_image_size)

    input_sketch: bpy.props.StringProperty(
        default='sketch', description='The sketch image on which to perform the inference')
    input_sketch: bpy.props.StringProperty(
        default='sketch', description='The sketch image on which to perform the inference', set=set_input_sketch, get=get_input_sketch)

    sample_preview: bpy.props.EnumProperty(
        items=enum_sample_preview, set=set_sample_preview, get=get_sample_preview)


class SketchGen(bpy.types.Operator):
    bl_idname = "terraindm.sketch_gen"
    bl_label = "Generate terrain"

    def execute(self, context):
        # Run on button press
        sketch.start_thread_gen()

        return {'FINISHED'}


class RandomSeed(bpy.types.Operator):
    bl_idname = "terraindm.random_seed"
    bl_label = "Select a random seed"

    def execute(self, context):
        context.scene.generation_props.seed = utils.random_max_int()
        return {'FINISHED'}


class StopGeneration(bpy.types.Operator):
    bl_idname = "terraindm.stop_generation"
    bl_label = "Stop the on-going generation"

    def execute(self, context):
        sketch.sketch_generator.stop()
        return {'FINISHED'}


class RealTimeStrokesGeneration(bpy.types.Operator):
    bl_idname = "terraindm.realtime_strokes_generation"
    bl_label = "Real time generation handler"

    _timer = None
    _old_img = None
    _old_is_equal = True
    # _prev_event_type = None

    _waiting_for_after_mousedown = False
    _last_refresh_time = 0

    def _update(self, context, real_time=False):
        current_time = time.time()
        # If real time, check if we should update based on previous refresh time
        if real_time:
            refresh_time = context.scene.display_settings_props.refresh_time
            if current_time - self._last_refresh_time <= refresh_time:
                return  # Do not update now

        input_sketch = context.scene.generation_props.input_sketch
        if not input_sketch:
            return

        # TODO: regenerate not with empty sketch or while loading
        current_img = utils.to_np_array(
            bpy.data.images[input_sketch], grayscale=False).astype(np.float32)
        is_equal = np.array_equal(self._old_img, current_img)

        # if not self._old_is_equal and is_equal: # Interactive
        if not is_equal:  # real-time
            sketch.start_thread_gen()
        self._old_img = current_img
        self._old_is_equal = is_equal

        self._last_refresh_time = current_time

    def modal(self, context, event):
        if context.object is not None and context.object.mode == 'TEXTURE_PAINT':
            if context.scene.generation_props.update_real_time:
                # Try update on any modal update. Since real_time=True,
                # we do a separate check based on previous update time
                self._update(context, real_time=True)

            if event.type != 'TIMER':
                if self._waiting_for_after_mousedown:  # Now process
                    if not context.scene.generation_props.update_real_time:
                        self._update(context)  # Interactive
                    self._waiting_for_after_mousedown = False

                if event.type == 'LEFTMOUSE':
                    self._waiting_for_after_mousedown = True

        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager

        # TODO make this a property
        self._timer = wm.event_timer_add(1/30, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)


@persistent
def load_realtime_stroke_generation_handler(_):
    bpy.ops.terraindm.realtime_strokes_generation()
