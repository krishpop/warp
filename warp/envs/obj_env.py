from warp.envs import WarpEnv
from warp.envs import builder_utils as bu
from warp.envs.common import ObjectType
from warp.envs.environment import RenderMode, IntegratorType, run_env


class ObjectEnv(WarpEnv):
    def __init__(
        self,
        num_envs,
        num_obs,
        num_act,
        episode_length,
        seed=0,
        no_grad=True,
        render=True,
        stochastic_init=False,
        device="cuda",
        render_mode=RenderMode.OPENGL,
        stage_path=None,
        object_type: ObjectType = ObjectType.SPRAY_BOTTLE,
    ):
        env_name = object_type.name + "Env"
        super().__init__(
            num_envs,
            num_obs,
            num_act,
            episode_length,
            seed,
            no_grad,
            render,
            stochastic_init,
            device,
            env_name,
            render_mode,
            stage_path,
        )

        self.object_type = object_type
        # self.joint_type = joint_type

        self.num_joint_q = 2
        self.num_joint_qd = 2

        self.init_sim()
        self.setup_autograd_vars()
        if self.use_graph_capture:
            self.graph_capture_params["bwd_model"].joint_attach_ke = self.joint_attach_ke
            self.graph_capture_params["bwd_model"].joint_attach_kd = self.joint_attach_kd

        self.simulate_params["ag_return_body"] = self.ag_return_body

    def create_articulation(self, builder):
        self.object_model = bu.OBJ_MODELS[self.object_type]()
        self.object_model.create_articulation(builder)
        self.start_pos = self.object_model.base_pos
        self.start_ori = self.object_model.base_ori


if __name__ == "__main__":
    run_env(lambda: ObjectEnv(5, 1, 1, 1000))
