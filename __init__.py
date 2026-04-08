# This file serves as the entry point for the blender add-on.

import bpy
import os

# Allow direct execution
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa
sys.path.insert(0, os.getcwd())  # noqa
from .app import settings, panels, operators


bl_info = {
    "name": "terrain-dm",
    "author": "Simon Perche and Joshua Lochner",
    "description": "",
    "blender": (3, 0, 0),
    "location": "View3D",
    "warning": "",
    "category": "Terrain generator"
}

classes = (
    settings.TerrainDMPreferences,

    # Order define the panels' order in the UI
    panels.MainPanel,
    panels.GenerationPanel,
    panels.BrushSelectionPanel,
    panels.ExtractionPanel,
    panels.StylePanel,
    panels.DisplaySettingPanel,
    panels.MiscellaneousPanel,

    operators.generation.GenerationProperty,
    operators.generation.SketchGen,
    operators.generation.StopGeneration,
    operators.generation.RealTimeStrokesGeneration,
    operators.generation.RandomSeed,

    operators.brushes.BrushesProperty,
    operators.brushes.BrushRivers,
    operators.brushes.BrushMountains,
    operators.brushes.BrushCliffs,
    operators.brushes.BrushFlat,
    operators.brushes.SketchErase,
    operators.brushes.BrushEraser,

    operators.extraction.ExtractionProperty,
    operators.extraction.ExtractSketch,

    operators.style.StyleProperty,
    operators.style.LoadStyle,
    operators.style.MountainPreview,
    operators.style.HillsPreview,
    operators.style.FlatPreview,
    operators.style.CliffsPreview,
    operators.style.SelectFlatStyle,
    operators.style.SelectMountainStyle,
    operators.style.SelectHillsStyle,
    operators.style.SelectCliffsStyle,

    operators.display_settings.DisplaySettingsProperty,

    operators.miscellaneous.LogFolder,
)


def register():
    operators.style.register()

    bpy.app.handlers.load_post.append(
        operators.generation.load_realtime_stroke_generation_handler)

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.brushes_props = bpy.props.PointerProperty(
        type=operators.brushes.BrushesProperty)
    bpy.types.Scene.styles_props = bpy.props.PointerProperty(
        type=operators.style.StyleProperty)
    bpy.types.Scene.generation_props = bpy.props.PointerProperty(
        type=operators.generation.GenerationProperty)
    bpy.types.Scene.display_settings_props = bpy.props.PointerProperty(
        type=operators.display_settings.DisplaySettingsProperty)
    bpy.types.Scene.extraction_props = bpy.props.PointerProperty(
        type=operators.extraction.ExtractionProperty)


def unregister():
    bpy.app.handlers.load_post.remove(
        operators.generation.load_realtime_stroke_generation_handler)

    del bpy.types.Scene.brushes_props
    del bpy.types.Scene.styles_props
    del bpy.types.Scene.generation_props
    del bpy.types.Scene.extraction_props

    operators.style.unregister()

    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
