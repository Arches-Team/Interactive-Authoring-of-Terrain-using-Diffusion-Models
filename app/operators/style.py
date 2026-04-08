import os

import bpy
from bpy_extras.io_utils import ImportHelper
from bpy.types import WindowManager
from bpy.props import StringProperty, EnumProperty
import bpy.utils.previews

from .. import utils
from ..generation import sketch

class LoadStyle(bpy.types.Operator, ImportHelper): 
    bl_idname = 'terraindm.load_style'
    bl_label = 'Load style'
    filter_glob: bpy.props.StringProperty(default='*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp', options={'HIDDEN'}) 

    def execute(self, context):
        bpy_img = bpy.data.images.load(self.filepath, check_existing=True)
        bpy_img.name = 'style'

        return {'FINISHED'}

preview_collections = {}
preview_props = {}
# Must match folder names
STYLE_NAMES = ["cliffs", "flat", "hills", "mountains"]
STYLE_PARENT_DIRECTORY = os.path.join(utils.get_filepath_in_package('images'), 'styles/')

class StyleProperty(bpy.types.PropertyGroup):
    mountains_selected: bpy.props.BoolProperty(default=True)
    cliffs_selected: bpy.props.BoolProperty(default=False)
    flat_selected: bpy.props.BoolProperty(default=False)
    hills_selected: bpy.props.BoolProperty(default=False)


class StylePreview():
    style_name = None
    displace_strength = 0.5
    
    def execute(self, context):
        bpy.ops.screen.userpref_show("INVOKE_DEFAULT")

        # Change area type
        area = context.window_manager.windows[-1].screen.areas[0]
        area.type = "VIEW_3D"
        space = next(s for s in area.spaces if s.type == 'VIEW_3D')
        space.shading.type = 'MATERIAL'

        # TODO: keep more than one?
        # Cannot call a callback on the closing of screen.
        # To reduce the memory footprint, only one object is kept at the same time
        collection_name = 'style_tmp'
        utils.delete_collection(collection_name)
        
        #TODO: not necessarily active object
        active_obj = bpy.context.active_object
        copy = utils.deepcopy_object(active_obj, copy_mat=True, show_in_3dview=True, collection_name=collection_name)
        copy.location.x += 2500.
        space.region_3d.view_location[0] += 2500.

        # for node in copy.active_material.node_tree.nodes:
        #     if node.bl_idname == 'ShaderNodeOutputMaterial':
        #         print(node.from_node)

        output_node = copy.active_material.node_tree.nodes.get('Material Output')
        last_node = output_node.inputs['Surface'].links[0].from_node
        last_node.inputs[0].default_value = 1.

        for modifier in copy.modifiers:
            if modifier.type == "DISPLACE":
                wm = context.window_manager
                style = bpy.data.images.load(os.path.abspath(eval(f'wm.preview_{self.style_name}')), check_existing=True)
                modifier.texture = modifier.texture.copy()
                modifier.texture.image = style
                
                modifier.strength = self.displace_strength
                modifier.show_viewport = True

        return {'FINISHED'}


class MountainPreview(bpy.types.Operator, StylePreview): 
    bl_idname = "terraindm.mountain_preview" 
    bl_label = "Mountain preview window"

    displace_strength = 0.5

    def __init__(self):
        super().__init__()
        self.style_name = 'mountains'
    
    def execute(self, context):
        return super().execute(context)


class HillsPreview(bpy.types.Operator, StylePreview): 
    bl_idname = "terraindm.hills_preview" 
    bl_label = "Hills preview window"

    displace_strength = 0.3

    def __init__(self):
        super().__init__()
        self.style_name = 'hills'
    
    def execute(self, context):
        return super().execute(context)


class FlatPreview(bpy.types.Operator, StylePreview): 
    bl_idname = "terraindm.flat_preview" 
    bl_label = "Flat preview window"

    displace_strength = 0.2

    def __init__(self):
        super().__init__()
        self.style_name = 'flat'
    
    def execute(self, context):
        return super().execute(context)


class CliffsPreview(bpy.types.Operator, StylePreview): 
    bl_idname = "terraindm.cliffs_preview" 
    bl_label = "Cliffs preview window"

    displace_strength = 0.2

    def __init__(self):
        super().__init__()
        self.style_name = 'cliffs'
    
    def execute(self, context):
        return super().execute(context)


def replace_style(context, preview_name):
     # Needed for the 'eval' function
    wm = context.window_manager
    bpy_img = bpy.data.images.load(os.path.abspath(eval(f'wm.preview_{preview_name}')), check_existing=True)
    bpy_img.colorspace_settings.name = 'Non-Color'

    if 'style' in bpy.data.images:
        style_img = bpy.data.images['style']
        style_img.reload()
        style_img.filepath = bpy_img.filepath
    else:
        bpy_img.name = "style"


class SelectStyle():
    _style = None

    def execute(self, context):
        replace_style(context, self._style)

        self._unselect_all(context)
        exec(f'context.scene.styles_props.{self._style}_selected = True')

        sketch.start_thread_gen()

        return {'FINISHED'}

    def _unselect_all(self, context):
        props = context.scene.styles_props
        props.mountains_selected = False
        props.cliffs_selected = False
        props.flat_selected = False
        props.hills_selected = False


class SelectMountainStyle(bpy.types.Operator, SelectStyle): 
    bl_idname = "terraindm.select_mountain_style" 
    bl_label = "Select mountain style"

    _style = 'mountains'

    def execute(self, context):
        return super().execute(context)


class SelectFlatStyle(bpy.types.Operator, SelectStyle): 
    bl_idname = "terraindm.select_flat_style" 
    bl_label = "Select flat style"

    _style = 'flat'

    def execute(self, context):
        return super().execute(context)


class SelectHillsStyle(bpy.types.Operator, SelectStyle): 
    bl_idname = "terraindm.select_hills_style" 
    bl_label = "Select hills style"

    _style = 'hills'

    def execute(self, context):
        return super().execute(context)


class SelectCliffsStyle(bpy.types.Operator, SelectStyle): 
    bl_idname = "terraindm.select_cliffs_style" 
    bl_label = "Select cliffs style"

    _style = 'cliffs'

    def execute(self, context):
        return super().execute(context)
        

def enum_previews_from_directory_items(context, directory, col_name):
    """EnumProperty callback"""
    enum_items = []

    if context is None:
        return enum_items

    wm = context.window_manager

    pcoll = preview_collections[col_name]

    if directory == pcoll.preview_dir:
        return pcoll.preview

    render_name = 'render.jpg'
    style_name = 'style.png'

    cpt = 0
    for root, dirs, files in os.walk(directory):
        render_file = os.path.abspath(os.path.join(root, render_name))
        dem_file = os.path.abspath(os.path.join(root, style_name))

        if os.path.isfile(render_file) and os.path.isfile(dem_file):
            icon = pcoll.get(render_file)
            if not icon:
                thumb = pcoll.load(render_file, render_file, 'IMAGE')
            else:
                thumb = pcoll[render_file]
            enum_items.append((dem_file, render_file, "", thumb.icon_id, cpt))
            cpt += 1

    pcoll.preview = enum_items
    pcoll.preview_dir = directory
    return pcoll.preview


# May be possible to reduce redundancy with global variables

def enum_preview_mountains(self, context):
    return enum_previews_from_directory_items(context, context.window_manager.preview_mountains_dir, 'mountains')


def update_preview_mountains(self, context):
    if context.scene.styles_props.mountains_selected:
        replace_style(context, 'mountains')


def enum_preview_flat(self, context):
    return enum_previews_from_directory_items(context, context.window_manager.preview_flat_dir, 'flat')


def update_preview_flat(self, context):
    if context.scene.styles_props.flat_selected:
        replace_style(context, 'flat')


def enum_preview_cliffs(self, context):
    return enum_previews_from_directory_items(context, context.window_manager.preview_cliffs_dir, 'cliffs')


def update_preview_cliffs(self, context):
    if context.scene.styles_props.cliffs_selected:
        replace_style(context, 'cliffs')


def enum_preview_hills(self, context):
    return enum_previews_from_directory_items(context, context.window_manager.preview_hills_dir, 'hills')


def update_preview_hills(self, context):
    if context.scene.styles_props.hills_selected:
        replace_style(context, 'hills')


# Called in register() to initialize previews
def register():
    for style_folder in STYLE_NAMES:
        # Maybe it's better to write each style
        exec(f'WindowManager.preview_{style_folder}_dir = StringProperty(name="Folder name", subtype="DIR_PATH", default=os.path.join(STYLE_PARENT_DIRECTORY, style_folder))')
        exec(f'WindowManager.preview_{style_folder} = EnumProperty(items=enum_preview_{style_folder}, update=update_preview_{style_folder})')

        pcoll = bpy.utils.previews.new()
        pcoll.preview_dir = ""
        pcoll.preview = ()

        preview_collections[style_folder] = pcoll

def unregister():
    for style_folder in STYLE_NAMES:
        exec(f'del WindowManager.preview_{style_folder}')

    for pcoll in preview_collections.values():
        bpy.utils.previews.remove(pcoll)
    preview_collections.clear()