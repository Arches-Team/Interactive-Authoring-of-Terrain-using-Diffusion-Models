import bpy

from .. import utils


def update_brush_size(self, context):
    context.tool_settings.unified_paint_settings.size = 3 if context.scene.brushes_props.brush_large_feature else 1


class BrushesProperty(bpy.types.PropertyGroup):
    brush_large_feature: bpy.props.BoolProperty(default=False, description='Paint large feature', update=update_brush_size)
    brush_rivers_selected: bpy.props.BoolProperty(default=False)
    brush_mountains_selected: bpy.props.BoolProperty(default=False)
    brush_cliffs_selected: bpy.props.BoolProperty(default=False)
    brush_flat_selected: bpy.props.BoolProperty(default=False)
    brush_eraser_selected: bpy.props.BoolProperty(default=False)


class Brushes():
    def __init__(self):
        self.brush_name = 'TerrainBrush'
        self.color = None

    def execute(self, context):
        brush = self._create_brush_if_not_exists()
        self._set_mode(brush)
        
        context.tool_settings.unified_paint_settings.color = self.color
        context.tool_settings.image_paint.brush = brush
        brush.use_paint_antialiasing = False
        context.tool_settings.proportional_edit_falloff = 'CONSTANT'
        update_brush_size(self, context)

        return {'FINISHED'}

    def _create_brush_if_not_exists(self):
        terrain_brush = [b for b in bpy.data.brushes if b.name == self.brush_name]
        if len(terrain_brush) == 0:
            terrain_brush = bpy.data.brushes.new(self.brush_name)
        else:
            terrain_brush = terrain_brush[0]

        terrain_brush.blend = 'ADD'
        terrain_brush.stroke_method = 'SPACE'
        terrain_brush.spacing = 1
        terrain_brush.curve_preset = 'CUSTOM'
        for point in terrain_brush.curve.curves[0].points:
            point.location = (0., 1.)
        terrain_brush.curve.curves[0].points[-2].location = (0.65, 1.)
        terrain_brush.curve.curves[0].points[-1].location = (0.651, 0.)
        terrain_brush.curve.update()
        
        return terrain_brush

    def _unselect_all(self, context):
        props = context.scene.brushes_props
        props.brush_rivers_selected = False
        props.brush_mountains_selected = False
        props.brush_cliffs_selected = False
        props.brush_flat_selected = False
        props.brush_eraser_selected = False

    def _set_mode(self, brush):
        brush.blend = 'ADD'
        

class BrushRivers(Brushes, bpy.types.Operator):
    bl_idname = 'terraindm.brushes_rivers'
    bl_label = 'Rivers brush'

    def __init__(self):
        super().__init__()
        self.color = (0., .99, 0.)

    def execute(self, context):
        self._unselect_all(context)
        context.scene.brushes_props.brush_rivers_selected = True
        return super().execute(context)


class BrushMountains(Brushes, bpy.types.Operator):
    bl_idname = 'terraindm.brushes_mountains'
    bl_label = 'Mountains brush'

    def __init__(self):
        super().__init__()
        self.color = (.99, 0., 0.)

    def execute(self, context):
        self._unselect_all(context)
        context.scene.brushes_props.brush_mountains_selected = True
        return super().execute(context)
        

class BrushCliffs(Brushes, bpy.types.Operator):
    bl_idname = 'terraindm.brushes_cliffs'
    bl_label = 'Cliffs brush'

    def __init__(self):
        super().__init__()
        self.color = (0., 0., .99)

    def execute(self, context):
        self._unselect_all(context)
        context.scene.brushes_props.brush_cliffs_selected = True
        return super().execute(context)


class BrushFlat(Brushes, bpy.types.Operator):
    bl_idname = 'terraindm.brushes_flat'
    bl_label = 'Flat area brush'

    def __init__(self):
        super().__init__()
        self.color = (.75, .75, .75)

    def execute(self, context):
        self._unselect_all(context)
        context.scene.brushes_props.brush_flat_selected = True
        return super().execute(context)


class BrushEraser(Brushes, bpy.types.Operator):
    bl_idname = 'terraindm.brushes_eraser'
    bl_label = 'Erase brush'

    def __init__(self):
        super().__init__()
        self.color = (0., 0., 0.)

    def execute(self, context):
        self._unselect_all(context)
        context.scene.brushes_props.brush_eraser_selected = True
        ret = super().execute(context)
        context.tool_settings.unified_paint_settings.size = 25
        return ret
        

    def _set_mode(self, brush):
        brush.blend = 'MIX'


class SketchErase(Brushes, bpy.types.Operator):
    bl_idname = 'terraindm.sketch_clear'
    bl_label = 'Erase the sketch'

    def execute(self, context):
        #TODO: use the prop
        sketch_img = bpy.data.images[context.scene.generation_props.input_sketch]
        pixels = [0, 0, 0, 1.0]*(int(len(sketch_img.pixels)/4))
        sketch_img.pixels[:] = pixels 

        utils.update_2d_3d_views_img(context.scene.generation_props.input_sketch)

        return {'FINISHED'}
