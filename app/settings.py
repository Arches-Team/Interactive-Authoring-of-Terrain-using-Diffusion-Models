import bpy


ADDON_NAME = 'terrain_dm'


class TerrainDMPreferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_NAME


class PanelSettings:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_options = {"DEFAULT_CLOSED"}


class TerrainDMProperty(bpy.types.PropertyGroup):
    # Sketch extraction
    sketch_extraction_range: bpy.props.FloatProperty(default=500, min=10, max=8000, description='Terrain range in meters')
    sketch_extraction_detailed: bpy.props.FloatProperty(default=0.5, min=0., max=1., description='How much the sketch is detailed in percentage. 1 as much as possible, 0 no sketch at all')


