import os
from utils.common import ObjectType
from utils.builder import OBJ_MODELS

def main():
    run_env_str = 'python3 run_task.py task=hand_object task.env.object_type="${{object:{}}}" alg.name=zero task.env.object_id={} num_rollouts=1 num_steps=2'

    for obj_type in ['scissors', 'bottle', 'dispenser', 'pliers']:
        for obj_id in OBJ_MODELS[ObjectType[obj_type.upper()]].keys():
            print('Running:', run_env_str.format(obj_type, obj_id))
            try:
                os.system(run_env_str.format(obj_type, obj_id))
            except KeyboardInterrupt:
                print('ending early')
                return

if __name__ == '__main__':
    main()

