import bpy

from . import settings


# Useful for children of main panel
MAIN_PANEL_NAME = 'TERRAINDM_PT_MainPanel'
GENERATION_PANEL_NAME = 'TERRAINDM_PT_GenerationPanel'
BRUSHES_PANEL_NAME = 'TERRAINDM_PT_BrushSelectionPanel'
EXTRACTION_PANEL_NAME = 'TERRAINDM_PT_ExtractionPanel'
STYLE_PANEL_NAME = 'TERRAINDM_PT_StylePanel'
DISPLAY_SETTINGS_PANEL_NAME = 'TERRAINDM_PT_DisplaySettingsPanel'
MISCELLANEOUS_PANEL_NAME = 'TERRAINDM_PT_MiscellaneousPanel'


class MainPanel(settings.PanelSettings, bpy.types.Panel):
    bl_idname = MAIN_PANEL_NAME
    bl_label = 'Terrain Diffusion'
    bl_options = {"HEADER_LAYOUT_EXPAND"}
    bl_category = 'Terrain'

    def draw(self, context):
        pass


class GenerationPanel(settings.PanelSettings, bpy.types.Panel):
    bl_idname = GENERATION_PANEL_NAME
    bl_label = 'Generation'
    bl_parent_id = MAIN_PANEL_NAME
    bl_options = {"HEADER_LAYOUT_EXPAND"}

    def draw(self, context):
        col = self.layout.column()
        generation_props = context.scene.generation_props

        is_sketch = generation_props.input_sketch != ''
        col.label(text='Inputs')
        box_inputs = col.box()
        if not is_sketch:  # or is_style:
            box_inputs.alert = True
            box_inputs.label(text='Please select a sketch')  # and a style.')
        box_inputs.prop_search(
            generation_props, 'input_sketch', bpy.data, 'images', text='Sketch')
        # box_inputs.prop_search(generation_props, 'input_style', bpy.data, 'images', text='Style')

        col.separator()

        col.prop(generation_props, 'nb_sample',
                 slider=True, text='Number of samples')
        col.prop(generation_props, 'sampling_steps',
                 slider=True, text='Number of sampling steps')
        col.prop(generation_props, 'guidance',
                 slider=True, text='Guidance scale')
        col.prop(generation_props, 'terrain_range', slider=True, text='Range')
        col.prop(generation_props, 'terrain_resolution',
                 slider=True, text='Resolution')

        col.label(text='Seed of the noise')
        box_seed = col.box()
        row_seed = box_seed.row().split(factor=.75)
        row_seed.prop(generation_props, 'seed', slider=False, text='Seed')
        row_seed.operator("terraindm.random_seed", text='Random')
        box_seed.prop(generation_props, 'random_seed',
                      text='Random seed when generating')
        row_seed.enabled = not generation_props.random_seed

        col.prop(generation_props, 'eta', slider=True, text='eta')
        col.prop(generation_props, 'fp16', slider=True,
                 text='16 bits float precision')
        col_size = col.column()
        col_size.prop(generation_props, 'image_size', text='Size')
        col.separator()

        col.label(text='Update')
        row = col.box().row()
        p = row.column()
        p.prop(generation_props, 'display_generation',
               text='Display generation')
        row.prop(generation_props, 'update_real_time', text='Real time')
        p.enabled = not generation_props.update_real_time

        # col.operator("terraindm.uncond_gen", text='Unconditional generation')
        col.separator()
        col_gen = col.column()
        col_gen.operator("terraindm.sketch_gen", text='Generate terrain')
        col_gen.operator("terraindm.stop_generation", text='Stop generation')

        # TODO: disable variations if only one sample
        col.separator()
        col.label(text='Variations')
        col.prop(generation_props, 'sample_preview', text='')

        # If no sketch is selected disable a few features
        if not is_sketch:
            col_size.enabled = False
            col_gen.enabled = False


class BrushSelectionPanel(settings.PanelSettings, bpy.types.Panel):
    bl_idname = BRUSHES_PANEL_NAME
    bl_label = 'Brushes'
    bl_parent_id = MAIN_PANEL_NAME

    def draw(self, context):
        col = self.layout.column()
        brushes_props = context.scene.brushes_props

        i = 'TRIA_RIGHT'
        col.prop(brushes_props, 'brush_large_feature',
                 text='Paint large feature')
        col.operator('terraindm.brushes_rivers', text='Rivers',
                     icon=i if brushes_props.brush_rivers_selected else 'NONE')
        col.operator('terraindm.brushes_mountains', text='Mountains',
                     icon=i if brushes_props.brush_mountains_selected else 'NONE')
        col.operator('terraindm.brushes_cliffs', text='Cliffs',
                     icon=i if brushes_props.brush_cliffs_selected else 'NONE')
        col.operator('terraindm.brushes_flat', text='Flat area',
                     icon=i if brushes_props.brush_flat_selected else 'NONE')
        col.operator('terraindm.brushes_eraser', text='Eraser',
                     icon=i if brushes_props.brush_eraser_selected else 'NONE')
        col.separator()
        col.operator('terraindm.sketch_clear', text='Clear the sketch')


class ExtractionPanel(settings.PanelSettings, bpy.types.Panel):
    bl_idname = EXTRACTION_PANEL_NAME
    bl_label = 'Sketch extraction'
    bl_parent_id = MAIN_PANEL_NAME

    def draw(self, context):
        extraction_props = context.scene.extraction_props

        col = self.layout.column()

        col.prop(extraction_props, 'range', slider=True, text='Range of input')
        col.prop(extraction_props, 'detailed_percentage',
                 slider=True, text='Details')
        col.operator('terraindm.extract_sketch',
                     text='Extract sketch from file')


class StylePanel(settings.PanelSettings, bpy.types.Panel):
    bl_idname = STYLE_PANEL_NAME
    bl_label = 'Style'
    bl_parent_id = MAIN_PANEL_NAME

    def draw(self, context):
        col = self.layout.column()
        styles_props = context.scene.styles_props
        radio_on = 'RADIOBUT_ON'
        radio_off = 'RADIOBUT_OFF'

        col.label(text='Predefined styles')

        first_row = col.row()
        box1 = first_row.box()
        box2 = first_row.box()

        box1.label(text='Mountains')
        box1.template_icon_view(context.window_manager,
                                "preview_mountains", scale=5, show_labels=False)
        mountains_buttons = box1.split(align=True, factor=0.8)
        mountains_buttons.operator(
            'terraindm.mountain_preview', text='Preview')
        mountains_buttons.operator('terraindm.select_mountain_style', text='',
                                   icon=radio_on if styles_props.mountains_selected else radio_off)

        box2.label(text='Cliffs')
        box2.template_icon_view(context.window_manager,
                                "preview_cliffs", scale=5, show_labels=False)
        cliffs_buttons = box2.split(align=True, factor=0.8)
        cliffs_buttons.operator('terraindm.cliffs_preview', text='Preview')
        cliffs_buttons.operator('terraindm.select_cliffs_style', text='',
                                icon=radio_on if styles_props.cliffs_selected else radio_off)

        second_row = col.row()
        box3 = second_row.box()
        box4 = second_row.box()

        box3.label(text='Plain')
        box3.template_icon_view(context.window_manager,
                                "preview_flat", scale=5, show_labels=False)
        flat_buttons = box3.split(align=True, factor=0.8)
        flat_buttons.operator('terraindm.flat_preview', text='Preview')
        flat_buttons.operator('terraindm.select_flat_style', text='',
                              icon=radio_on if styles_props.flat_selected else radio_off)

        box4.label(text='Hills')
        box4.template_icon_view(context.window_manager,
                                "preview_hills", scale=5, show_labels=False)
        hills_buttons = box4.split(align=True, factor=0.8)
        hills_buttons.operator('terraindm.hills_preview', text='Preview')
        hills_buttons.operator('terraindm.select_hills_style', text='',
                               icon=radio_on if styles_props.hills_selected else radio_off)

        col.separator()
        col.operator('terraindm.load_style', text='Load custom style')


class DisplaySettingPanel(settings.PanelSettings, bpy.types.Panel):
    bl_idname = DISPLAY_SETTINGS_PANEL_NAME
    bl_label = 'Display Settings'
    bl_parent_id = MAIN_PANEL_NAME

    def draw(self, context):

        display_settings_props = context.scene.display_settings_props
        col = self.layout.column()

        col.prop(display_settings_props, 'ignore_first_x_steps',
                 slider=True, text='Ignore first X generation steps')
        col.prop(display_settings_props, 'refresh_time',
                 slider=True, text='Refresh rate')


class MiscellaneousPanel(settings.PanelSettings, bpy.types.Panel):
    bl_idname = MISCELLANEOUS_PANEL_NAME
    bl_label = 'Miscellaneous'
    bl_parent_id = MAIN_PANEL_NAME

    def draw(self, context):
        col = self.layout.column()

        col.operator("terraindm.choose_log_folder",
                     text='Choose the log folder')
