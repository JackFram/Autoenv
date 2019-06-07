import envs.hyperparams as hyperparams
import os
from envs import build_env


# the normal function
def run(args):
    print("loading from:", args.params_filepath)
    print("saving to:", args.exp_name)

    # build components
    env, act_low, act_high = build_env.build_ngsim_env(args, vectorize=args.vectorize)

    obs = env.reset()
    for i in range(50):
        action = [1, 0.1]
        features, reward, terminal, infos = env.step(action)
        obs = features
        print(infos)
    return env

    # data = utils.load_data(
    #     args.expert_filepath,
    #     act_low=act_low,
    #     act_high=act_high,
    #     min_length=args.env_H + args.env_primesteps,
    #     clip_std_multiple=args.normalize_clip_std_multiple,
    #     ngsim_filename=args.ngsim_filename
    # )


# setup
if __name__ == '__main__':
    params = hyperparams.parse_args()
    run(params)

