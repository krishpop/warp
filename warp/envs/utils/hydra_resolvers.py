from omegaconf import OmegaConf


try:
    OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg in ["", None] else arg)
except ValueError:
    # already registered
    pass

try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    # already registered
    pass


def return_enum(x, y):
    from warp.envs.utils.common import ActionType, HandType, ObjectType, GoalType, RewardType

    return eval(x)[y.upper()]


OmegaConf.register_new_resolver("enum", return_enum)

from warp.envs.utils.common import ActionType, HandType, ObjectType, GoalType, RewardType
from warp.envs.environment import RenderMode

OmegaConf.register_new_resolver("object", lambda x: ObjectType[x.upper()])
OmegaConf.register_new_resolver("hand", lambda x: HandType[x.upper()])
OmegaConf.register_new_resolver("action", lambda x: ActionType[x.upper()])
OmegaConf.register_new_resolver("goal", lambda x: GoalType[x.upper()])
OmegaConf.register_new_resolver("reward", lambda x: RewardType[x.upper()])
OmegaConf.register_new_resolver("render", lambda x: RenderMode[x.upper()])
