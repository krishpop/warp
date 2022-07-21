import carb
import carb.settings
import omni.kit.menu.utils
import omni.usd
import omni.timeline
import warp as wp
from omni.kit.menu.utils import MenuItemDescription
from . import menu_common
from .common import log_error
import os
import imp
import webbrowser

SCRIPTS_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/scripts"))
SCENES_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/scenes"))

WARP_GETTING_STARTED_URL = "https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_warp.html"
WARP_DOCUMENTATION_URL = "https://nvidia.github.io/warp/"

class WarpMenu:

    def __init__(self):

        self._is_live = True

        editor_menu = omni.kit.ui.get_editor_menu()

        if editor_menu:

            # Disabled until we can do some more polish

            # self._example_DEM_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_DEM_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_DEM_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_MESH_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_MESH_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_MESH_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_NVDB_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_NVDB_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_NVDB_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_SIM_CLOTH_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_SIM_CLOTH_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_SIM_CLOTH_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_SIM_GRANULAR_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_SIM_GRANULAR_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_SIM_GRANULAR_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_SIM_NEO_HOOKEAN_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_SIM_NEO_HOOKEAN_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_SIM_NEO_HOOKEAN_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_SIM_PARTICLE_CHAIN_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_SIM_PARTICLE_CHAIN_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_SIM_PARTICLE_CHAIN_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_SIM_RIGID_CHAIN_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_SIM_RIGID_CHAIN_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_SIM_RIGID_CHAIN_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_SIM_RIGID_CONTACT_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_SIM_RIGID_CONTACT_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_SIM_RIGID_CONTACT_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_SIM_RIGID_FEM_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_SIM_RIGID_FEM_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_SIM_RIGID_FEM_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_SIM_RIGID_FORCE_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_SIM_RIGID_FORCE_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_SIM_RIGID_FORCE_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_SIM_RIGID_GYROSCOPIC_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_SIM_RIGID_GYROSCOPIC_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_SIM_RIGID_GYROSCOPIC_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_WAVE_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_WAVE_MENU_ITEM}", 
            #     lambda _, value: self._on_script_menu_click(menu_common.EXAMPLE_WAVE_SCRIPT),
            #     toggle=False, value=False
            # )
            # self._example_BROWSE_menu_item = editor_menu.add_item(
            #     f"Window/{menu_common.EXAMPLE_BROWSE_MENU_ITEM}", 
            #     lambda _, value: self._on_browse_scripts_click(),
            #     toggle=False, value=False
            # )

            self._scene_CLOTH_menu_item = editor_menu.add_item(
                f"Window/{menu_common.SCENE_CLOTH_MENU_ITEM}", 
                lambda _, value: self._on_scene_menu_click(menu_common.SCENE_CLOTH),
                toggle=False, value=False
            )
            self._scene_DEFORM_menu_item = editor_menu.add_item(
                f"Window/{menu_common.SCENE_DEFORM_MENU_ITEM}", 
                lambda _, value: self._on_scene_menu_click(menu_common.SCENE_DEFORM),
                toggle=False, value=False
            )
            self._scene_PARTICLES_menu_item = editor_menu.add_item(
                f"Window/{menu_common.SCENE_PARTICLES_MENU_ITEM}", 
                lambda _, value: self._on_scene_menu_click(menu_common.SCENE_PARTICLES),
                toggle=False, value=False
            )
            self._scene_WAVE_menu_item = editor_menu.add_item(
                f"Window/{menu_common.SCENE_WAVE_MENU_ITEM}", 
                lambda _, value: self._on_scene_menu_click(menu_common.SCENE_WAVE),
                toggle=False, value=False
            )
            self._scene_MARCHING_menu_item = editor_menu.add_item(
                f"Window/{menu_common.SCENE_MARCHING_MENU_ITEM}", 
                lambda _, value: self._on_scene_menu_click(menu_common.SCENE_MARCHING),
                toggle=False, value=False
            )            
            self._scene_BROWSE_menu_item = editor_menu.add_item(
                f"Window/{menu_common.SCENE_BROWSE_MENU_ITEM}", 
                lambda _, value: self._on_browse_scenes_click(),
                toggle=False, value=False, priority=100     # set priority to insert a separator line (omni.kit.ui)
            )

            self._help_GETTING_STARTED_menu_item = editor_menu.add_item(
                "Window/Warp/Help/Getting Started", 
                lambda _, value: self._on_getting_started_click(),
                toggle=False, value=False
            )
            self._help_DOCUMENTATION_menu_item = editor_menu.add_item(
                "Window/Warp/Help/Documentation", 
                lambda _, value: self._on_documentation_click(),
                toggle=False, value=False
            )

        self._update_event_stream = omni.kit.app.get_app_interface().get_update_event_stream()
        self._stage_event_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self._on_stage_event)

    def shutdown(self):
        self._example_DEM_menu_item = None
        self._example_MESH_menu_item = None
        self._example_NVDB_menu_item = None
        self._example_SIM_CLOTH_menu_item = None
        self._example_SIM_GRANULAR_menu_item = None
        self._example_SIM_NEO_HOOKEAN_menu_item = None
        self._example_SIM_PARTICLE_CHAIN_menu_item = None
        self._example_SIM_RIGID_CHAIN_menu_item = None
        self._example_SIM_RIGID_CONTACT_menu_item = None
        self._example_SIM_RIGID_FEM_menu_item = None
        self._example_SIM_RIGID_FORCE_menu_item = None
        self._example_SIM_RIGID_GYROSCOPIC_menu_item = None
        self._example_WAVE_menu_item = None
        self._example_BROWSE_menu_item = None

        self._scene_CLOTH_menu_item = None
        self._scene_DEFORM_menu_item = None
        self._scene_PARTICLES_menu_item = None
        self._scene_WAVE_menu_item = None
        self._scene_BROWSE_menu_item = None

        self._help_GETTING_STARTED_menu_item = None
        self._help_DOCUMENTATION_menu_item = None

        self._example = None
        self._update_event_sub = None
        self._update_event_stream = None
        self._stage_event_sub = None

    def _on_update(self, event):

        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing() and self._example is not None:
            with wp.ScopedDevice("cuda:0"):
                self._example.update()
                self._example.render(is_live=self._is_live)

    def _on_stage_event(self, event):

        if event.type == int(omni.usd.StageEventType.CLOSED):
            self._refresh_example()
        if event.type == int(omni.usd.StageEventType.OPENED):
            self._refresh_example()

    def _reset_example(self):

        if self._example is not None:
            stage = omni.usd.get_context().get_stage()
            stage.GetRootLayer().Clear()
            with wp.ScopedDevice("cuda:0"):
                self._example.init(stage)
                self._example.render(is_live=self._is_live)

    def _refresh_example(self):

        self._example = None
        self._update_event_sub = None

    def _on_script_menu_click(self, script_name):

        def new_stage():
            
            new_stage = omni.usd.get_context().new_stage()
            if new_stage:
                stage = omni.usd.get_context().get_stage()
            else:
                log_error("Could not open new stage")
                return

            import_path = os.path.normpath(os.path.join(SCRIPTS_PATH, script_name))

            module = imp.load_source(script_name, import_path)
            self._example = module.Example()

            if self._example is None:
                log_error("Problem loading example module")
                return
            if not hasattr(self._example, 'init'):
                log_error("Example missing init() function")
                return
            if not hasattr(self._example, 'update'):
                log_error("Example missing update() function")
                return
            if not hasattr(self._example, 'render'):
                log_error("Example missing render() function")
                return

            with wp.ScopedDevice("cuda:0"):
                self._example.init(stage)
                self._example.render(is_live=self._is_live)

            # focus camera
            omni.usd.get_context().get_selection().set_selected_prim_paths([stage.GetDefaultPrim().GetPath().pathString], False)
            viewport_window = omni.kit.viewport_legacy.get_viewport_interface().get_viewport_window()
            if viewport_window:
                viewport_window.focus_on_selected()
            omni.usd.get_context().get_selection().clear_selected_prim_paths()

            self._update_event_sub = self._update_event_stream.create_subscription_to_pop(self._on_update)

        omni.kit.window.file.prompt_if_unsaved_stage(new_stage)


    def _on_scene_menu_click(self, scene_name):
        
        def new_stage():
            stage_path = os.path.normpath(os.path.join(SCENES_PATH, scene_name))
            omni.usd.get_context().open_stage(stage_path)

        omni.kit.window.file.prompt_if_unsaved_stage(new_stage)

    def _on_browse_scripts_click(self):
        os.startfile(SCRIPTS_PATH)

    def _on_browse_scenes_click(self):
        os.startfile(SCENES_PATH)

    def _on_getting_started_click(self, *_):
        webbrowser.open(WARP_GETTING_STARTED_URL)

    def _on_documentation_click(self, *_):
        webbrowser.open(WARP_DOCUMENTATION_URL)