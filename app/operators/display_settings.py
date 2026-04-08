import bpy


class DisplaySettingsProperty(bpy.types.PropertyGroup):
    ignore_first_x_steps: bpy.props.IntProperty(
        default=0, min=0, max=10,
        description='Do not display first X generation steps (improves coherence when authoring in real-time)'
    )
    refresh_time: bpy.props.FloatProperty(
        default=0.1,
        min=1/20,  # 20 fps
        max=1,  # 1 fps
        description='Do not display first X generation steps (improves coherence when authoring in real-time)'
    )
