import bpy
from bpy_extras.io_utils import ImportHelper

from ..utils import Logger

class LogFolder(bpy.types.Operator, ImportHelper): 
    bl_idname = "terraindm.choose_log_folder" 
    bl_label = "Choose the log folder"
    use_filter_folder = True

    def execute(self, context):
        Logger().start(self.filepath)
        if not Logger().is_init():
            self.report({'ERROR'}, 'Please choose folder with write permission (on the desktop for example)')
        
        return {'FINISHED'}