import argparse
import h5py
import multiprocessing as mp
import numpy as np
import os
import sys
import tensorflow as tf
import time
# import random
import pickle
import julia

backend = 'TkAgg'
import matplotlib

matplotlib.use(backend)
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
from contexttimer import Timer

import hgail.misc.simulation
import hgail.misc.utils
import algorithms.utils

from envs import hyperparams, utils, build_env

from envs.utils import str2bool
from utils.math_utils import classify_traj
from algorithms.AGen import rls, validate_utils
from preprocessing.clean_holo import clean_data, csv2txt, create_lane
from preprocessing.extract_feature import extract_ngsim_features
from src.trajdata import convert_raw_ngsim_to_trajdatas
# import pdb
import math
import tqdm
import torch

plt.style.use("ggplot")

# TODO: change this accordingly
EGO_START_FRAME = 1106
N_VEH = 1
EGO_ID = 1978
DATA_INDEX = [96]
N_ITERATION = 1
MAX_STEP = 150
TOTAL_STEP = 0

Veh_counter = 0

def online_adaption(
        env,
        policy,
        max_steps,
        obs,
        mean,
        render=False,
        env_kwargs=dict(),
        lbd=0.99,
        adapt_steps=1,
        nids=1,
        trajinfos=None):

    if len(obs.shape) == 2:
        obs = np.expand_dims(obs, axis=0)
        mean = np.expand_dims(mean, axis=0)
    assert trajinfos is not None
    # theta = np.load('./data/theta.npy')  # TODO: change the file path
    # theta = np.mean(theta)
    #
    # print("original theta: {}".format(theta))

    policy_fc_weight = np.array(policy.mean_network.fc.weight.data.cpu())
    policy_fc_bias = np.array(policy.mean_network.fc.bias.data.cpu()).reshape((2, 1))
    new_theta = np.concatenate([policy_fc_weight, policy_fc_bias], axis=1)
    new_theta = np.mean(new_theta)

    # print("new theta: {}".format(new_theta))

    ego_start_frame = trajinfos[env_kwargs['egoid']]['ts']
    maxstep = trajinfos[env_kwargs['egoid']]['te'] - trajinfos[env_kwargs['egoid']]['ts'] - 52
    env_kwargs['start'] = ego_start_frame + 2
    x = env.reset(**env_kwargs)

    n_agents = x.shape[0]
    # print("Agent number: {}".format(n_agents))
    dones = [True] * n_agents
    predicted_trajs, adapnets = [], []
    policy.reset(dones)

    # max_steps = min(200, obs.shape[1] - primesteps - 2)
    print("max steps")
    print(maxstep)
    mean = np.expand_dims(mean, axis=2)
    prev_hiddens = np.zeros([n_agents, 64])

    param_length = 65 if adapt_steps == 1 else 195

    for i in range(n_agents):
        adapnets.append(rls.rls(lbd, new_theta, param_length, 2))

    # print(('Reset env Running time: %s Seconds' % (end_time - start_time)))
    lx = x
    error = []  # size is (maxstep, predict_span, n_agent) each element is a dict(dx: , dy: ,dist: )
    curve_error = []
    changeLane_error = []
    straight_error = []
    orig_traj_list = []
    pred_traj_list = []
    time_list = []
    for step in tqdm.tqdm(range(ego_start_frame, maxstep + ego_start_frame - 1)):

        a, a_info, hidden_vec = policy.get_actions_with_prev(obs[:, step, :], mean[:, step, :], prev_hiddens)
        # print(hidden_vec)
        if adapt_steps == 1:
            adap_vec = hidden_vec
        elif adapt_steps == 2:
            adap_vec = np.concatenate((hidden_vec, prev_hiddens, obs[:, step, :]), axis=1)
        else:
            print('Adapt steps can only be 1 and 2 for now.')
            exit(0)

        adap_vec = np.expand_dims(adap_vec, axis=1)

        for i in range(n_agents):
            for _ in range(N_ITERATION):
                adapnets[i].update(adap_vec[i], mean[i, step+1, :])
                adapnets[i].draw.append(adapnets[i].theta[6, 1])

        prev_actions, prev_hiddens = a, hidden_vec

        traj, error_per_step, time_info, orig_traj, pred_traj = prediction(env_kwargs, x, adapnets, env, policy,
                                                                           prev_hiddens, n_agents, adapt_steps, nids)
        print("Vehicle Counter: {}".format(Veh_counter))
        break
        traj_cat = classify_traj(orig_traj)

        if traj_cat != "invalid":
            error.append(error_per_step)
            orig_traj_list.append(orig_traj)
            pred_traj_list.append(pred_traj)
        if traj_cat == "curve":
            curve_error.append(error_per_step)
        elif traj_cat == "changeLane":
            changeLane_error.append(error_per_step)
        elif traj_cat == "straight":
            straight_error.append(error_per_step)
        if "20" in time_info.keys() and "50" in time_info.keys():
            time_list.append([time_info["20"], time_info["50"]])
        predicted_trajs.append(traj)
        d = np.stack([adapnets[i].draw for i in range(n_agents)])

        env_kwargs['start'] += 1
        lx = x
        x = env.reset(**env_kwargs)

    error_info = dict()
    error_info["overall"] = error
    error_info["curve"] = curve_error
    error_info["lane_change"] = changeLane_error
    error_info["straight"] = straight_error
    error_info["time_info"] = time_list
    error_info["orig_traj"] = orig_traj_list
    error_info["pred_traj"] = pred_traj_list
    print("\n\nVehicle id: {} Statistical Info:\n\n".format(env_kwargs['egoid']))
    print("Vehicle Counter: {}".format(Veh_counter))
    utils.print_error(error_info)

    return predicted_trajs, error_info


def prediction(env_kwargs, x, adapnets, env, policy, prev_hiddens, n_agents, adapt_steps, nids):
    traj = hgail.misc.simulation.Trajectory()
    predict_span = 400
    # predict_span = 50
    error_per_step = []  # size is (predict_span, n_agent) each element is a dict(dx: , dy: ,dist: )
    valid_data = True
    hi_speed_limit = 40
    lo_speed_limit = 10
    orig_trajectory = []
    pred_trajectory = []
    start_time = time.time()
    time_info = {}
    feature_array = np.zeros([0, 66])
    lane_array = []
    for j in range(predict_span):
        # if j == 0:
        #     print("feature {}".format(j), x)
        a, a_info, hidden_vec = policy.get_actions(x)

        if adapt_steps == 1:
            adap_vec = hidden_vec
        else:
            adap_vec = np.concatenate((hidden_vec, prev_hiddens, x), axis=1)

        means = np.zeros([n_agents, 2])
        log_std = np.zeros([n_agents, 2])
        for i in range(x.shape[0]):
            means[i] = adapnets[i].predict(np.expand_dims(adap_vec[i], 0))
            log_std[i] = np.log(np.std(adapnets[i].theta, axis=0))

        prev_hiddens = hidden_vec

        # rnd = np.random.normal(size=means.shape)
        actions = means
        # print("predict step: {}".format(j+1))
        nx, r, dones, e_info = env.step(actions)
        traj.add(x, actions, r, a_info, e_info)
        error_per_agent = []  # length is n_agent, each element is a dict(dx: , dy: ,dist: )

        for i in range(n_agents):
            assert n_agents == 1
            # print("orig x: ", e_info["orig_x"][i])
            # print("orig y: ", e_info["orig_y"][i])
            # print("orig v: ", e_info["orig_v"][i])
            # print("orig theta: ", e_info["orig_theta"][i])
            # print("predicted x: ", e_info["x"][i])
            # print("predicted y: ", e_info["y"][i])
            dx = abs(e_info["orig_x"][i] - e_info["x"][i])
            dy = abs(e_info["orig_y"][i] - e_info["y"][i])
            dist = math.hypot(dx, dy)
            # print("dist: ", dist)
            # print("{}-----> dx: {} dy: {} dist: {}".format(j, dx, dy, dist))
            error_per_agent.append(dist)
            orig_trajectory.append([e_info["orig_x"][i], e_info["orig_y"][i]])
            pred_trajectory.append([e_info["x"][i], e_info["y"][i]])

        error_per_step += error_per_agent
        x = nx
        feature_array = np.concatenate([feature_array, np.array(x)], axis=0)
        lane_array.append(e_info["lane_id"][0])
        end_time = time.time()
        if j == 19:
            time_info["20"] = end_time - start_time
        elif j == 49:
            time_info["50"] = end_time - start_time
        if any(dones):
            # break
            continue
    lane_array = np.array(lane_array)
    print(feature_array.shape, np.array(orig_trajectory).shape, lane_array.shape)
    global Veh_counter
    np.savez("./abu/{}.npz".format(Veh_counter), feature=feature_array, trajectory=np.array(orig_trajectory), lane_id=lane_array)
    print("Trajectory has been saved to ./abu/{}.npz".format(Veh_counter))
    Veh_counter += 1
    if Veh_counter == 100:
        exit(0)
    return traj.flatten(), error_per_step, time_info, orig_trajectory, pred_trajectory


def collect_trajectories(
        args,
        params,
        egoids,
        starts,
        error_dict,
        pid,
        env_fn,
        policy_fn,
        max_steps,
        use_hgail,
        random_seed,
        lbd,
        adapt_steps):
    print('env initialization args')
    print(args)
    env, trajinfos, _, _ = env_fn(args, n_veh=N_VEH, alpha=0.)
    # print(trajinfos[0])
    args.policy_recurrent = True
    policy = policy_fn(args, env, mode=1)
    if torch.cuda.is_available():
        policy = policy.cuda()
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # then load parameters
        if use_hgail:
            for i, level in enumerate(policy):
                level.algo.policy.set_param_values(params[i]['policy'])
            policy = policy[0].algo.policy
        else:
            policy_param_path = "./data/experiments/NGSIM-gail/imitate/model/policy_700.pkl"
            policy.load_param(policy_param_path)
            print("load policy param from: {}".format(policy_param_path))
            # policy.set_param_values(params['policy'])

        normalized_env = hgail.misc.utils.extract_normalizing_env(env)
        if normalized_env is not None:
            normalized_env._obs_mean = params['normalzing']['obs_mean']
            normalized_env._obs_var = params['normalzing']['obs_var']
            print(params['normalzing']['obs_mean'], params['normalzing']['obs_var'])

        # collect trajectories
        egoids = np.unique(egoids)
        nids = len(egoids)
        veh_2_index = {}
        if args.env_multiagent:
            data, index = validate_utils.get_multiagent_ground_truth(args.ngsim_filename, args.h5_filename)
            for i, idx in enumerate(index):
                veh_2_index[idx] = i
        else:
            data = validate_utils.get_ground_truth(args.ngsim_filename, args.h5_filename)
            sample = np.random.choice(data['observations'].shape[0], 2)

        kwargs = dict()
        # print(('Loading obs data Running time: %s Seconds' % (end_time - start_time)))
        if args.env_multiagent:
            # I add not because single simulation has no orig_x etc.
            # egoid = random.choice(egoids)
            trajinfos = trajinfos[0]
            error = {"overall": [],
                     "curve": [],
                     "lane_change": [],
                     "straight": [],
                     "time_info": [],
                     "orig_traj": [],
                     "pred_traj": []}
            for veh_id in trajinfos.keys():
                if trajinfos[veh_id]["te"] - trajinfos[veh_id]["ts"] <= 452:
                    continue
                if random_seed:
                    kwargs = dict(random_seed=random_seed + veh_id)
                print("egoid: {}, ts: {}, te: {}".format(veh_id, trajinfos[veh_id]["ts"], trajinfos[veh_id]["te"]))
                print("data index is {}".format(veh_2_index[veh_id]))
                kwargs['egoid'] = veh_id
                kwargs['traj_idx'] = 0

                traj, error_info = online_adaption(
                    env,
                    policy,
                    max_steps=max_steps,
                    obs=data['observations'][[veh_2_index[veh_id]], :, :],
                    mean=data['actions'][[veh_2_index[veh_id]], :, :],
                    env_kwargs=kwargs,
                    lbd=lbd,
                    adapt_steps=adapt_steps,
                    nids=nids,
                    trajinfos=trajinfos
                )
                print("Vehicle Counter: {}".format(Veh_counter))

                error["overall"] += error_info["overall"]
                error["curve"] += error_info["curve"]
                error["lane_change"] += error_info["lane_change"]
                error["straight"] += error_info["straight"]
                error["time_info"] += error_info["time_info"]
                error["orig_traj"] += error_info["orig_traj"]
                error["pred_traj"] += error_info["pred_traj"]
            error_dict.append(error)
        else:
            # for i in sample:
            for i, egoid in enumerate(egoids):
                sys.stdout.write('\rpid: {} traj: {} / {}\n'.format(pid, i, nids))
                index = veh_2_index[egoid]
                traj = online_adaption(
                    env,
                    policy,
                    max_steps=max_steps,
                    obs=data['observations'][index, :, :],
                    mean=data['actions'][index, :, :],
                    env_kwargs=dict(egoid=egoid, traj_idx=[0]),
                    lbd=lbd,
                    adapt_steps=adapt_steps,
                    nids=nids
                )
                # trajlist.append(traj)

    return error_dict


def parallel_collect_trajectories(
        args,
        params,
        egoids,
        starts,
        n_proc,
        env_fn=build_env.build_ngsim_env,
        max_steps=200,
        use_hgail=False,
        random_seed=None,
        lbd=0.99,
        adapt_steps=1):
    # build manager and dictionary mapping ego ids to list of trajectories

    tf_policy = False
    parallel = False
    # set policy function
    policy_fn = validate_utils.build_policy if tf_policy else algorithms.utils.build_policy

    # partition egoids
    proc_egoids = utils.partition_list(egoids, n_proc)
    if parallel:
        manager = mp.Manager()
        error_dict = manager.list()
        # pool of processes, each with a set of ego ids
        pool = mp.Pool(processes=n_proc)
        # print(('Creating parallel env Running time: %s Seconds' % (end_time - start_time)))
        # run collection
        results = []
        for pid in range(n_proc):
            res = pool.apply_async(
                collect_trajectories,
                args=(
                    args,
                    params,
                    proc_egoids[pid],
                    starts,
                    error_dict,
                    pid,
                    env_fn,
                    policy_fn,
                    max_steps,
                    use_hgail,
                    random_seed,
                    lbd,
                    adapt_steps
                )
            )
            results.append(res)
        [res.get() for res in results]
        pool.close()
    else:
        error_dict = []
        error_dict = collect_trajectories(
            args,
            params,
            proc_egoids[0],
            starts,
            error_dict,
            0,
            env_fn,
            policy_fn,
            max_steps,
            use_hgail,
            random_seed,
            lbd,
            adapt_steps
        )

    # wait for the processes to finish

    print("Vehicle Counter: {}".format(Veh_counter))

    # let the julia processes finish up
    time.sleep(10)
    return error_dict[0]


def single_process_collect_trajectories(
        args,
        params,
        egoids,
        starts,
        n_proc,
        env_fn=build_env.build_ngsim_env,
        max_steps=200,
        use_hgail=False,
        random_seed=None):
    '''
    This function for debugging purposes
    '''
    # build list to be appended to
    trajlist = []

    # set policy function
    policy_fn = build_env.build_hierarchy if use_hgail else validate_utils.build_policy
    tf.reset_default_graph()

    # collect trajectories in a single process
    collect_trajectories(
        args,
        params,
        egoids,
        starts,
        trajlist,
        n_proc,
        env_fn,
        policy_fn,
        max_steps,
        use_hgail,
        random_seed
    )
    return trajlist


def collect(
        egoids,
        starts,
        args,
        exp_dir,
        use_hgail,
        params_filename,
        n_proc,
        max_steps=200,
        collect_fn=parallel_collect_trajectories,
        random_seed=None,
        lbd = 0.99,
        adapt_steps=1):
    '''
    Description:
        - prepare for running collection in parallel
        - multiagent note: egoids and starts are not currently used when running
            this with args.env_multiagent == True
    '''
    # load information relevant to the experiment
    params_filepath = os.path.join(exp_dir, 'imitate/{}'.format(params_filename))
    params = hgail.misc.utils.load_params(params_filepath)
    # validation setup
    validation_dir = os.path.join(exp_dir, 'imitate', 'test')
    utils.maybe_mkdir(validation_dir)
    output_filepath = os.path.join(validation_dir, '{}_AGen_{}_{}.npz'.format(
        args.ngsim_filename.split('.')[0], adapt_steps, args.env_multiagent))

    with Timer():
        error = collect_fn(
            args,
            params,
            egoids,
            starts,
            n_proc,
            max_steps=max_steps,
            use_hgail=use_hgail,
            random_seed=random_seed,
            lbd=lbd,
            adapt_steps=adapt_steps
        )
    print("Vehicle Counter: {}".format(Veh_counter))
    return error

    # utils.write_trajectories(output_filepath, trajs)


def load_egoids(filename, args, n_runs_per_ego_id=10, env_fn=build_env.build_ngsim_env):
    offset = args.env_H + args.env_primesteps
    basedir = os.path.expanduser('~/Autoenv/data/')  # TODO: change the file path
    ids_filename = filename.replace('.txt', '-index-{}-ids.h5'.format(offset))
    print("ids_filename")
    print(ids_filename)
    ids_filepath = os.path.join(basedir, ids_filename)
    traj_num = 0
    if True:
        print("Creating ids file")
        # this should create the ids file
        env_fn(args)
        if not os.path.exists(ids_filepath):
            raise ValueError('file unable to be created, check args')
    ids = np.array(h5py.File(ids_filepath, 'r')['ids'].value)

    # we want to sample start times uniformly from the range of possible values
    # but we also want these start times to be identical for every model we
    # validate. So we sample the start times a single time, and save them.
    # if they exist, we load them in and reuse them
    start_times_filename = filename.replace('.txt', '-index-{}-starts.h5'.format(offset))
    start_times_filepath = os.path.join(basedir, start_times_filename)
    # check if start time filepath exists
    # if os.path.exists(start_times_filepath):
    if False:
        # load them in
        starts = np.array(h5py.File(start_times_filepath, 'r')['starts'].value)
    # otherwise, sample the start times and save them
    else:
        print("Creating starts file")
        ids_file = h5py.File(ids_filepath, 'r')
        ts = ids_file['ts'].value
        # subtract offset gives valid end points
        te = ids_file['te'].value
        length = np.array([e - s for (s, e) in zip(ts, te)])
        traj_num = length.sum()
        # write to file
        # starts_file = h5py.File(start_times_filepath, 'w')
        # starts_file.create_dataset('starts', data=starts)
        # starts_file.close()

    # create a dict from id to start time

    ids = np.tile(ids, n_runs_per_ego_id)
    return ids, traj_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validation settings')
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--exp_dir', type=str, default='./data/experiments/NGSIM-gail')
    parser.add_argument('--params_filename', type=str, default='itr_700.npz')
    parser.add_argument('--n_runs_per_ego_id', type=int, default=1)
    parser.add_argument('--use_hgail', type=str2bool, default=False)
    parser.add_argument('--use_multiagent', type=str2bool, default=False)
    parser.add_argument('--n_multiagent_trajs', type=int, default=10000)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--n_envs', type=int, default=None)
    parser.add_argument('--remove_ngsim_vehicles', type=str2bool, default=False)
    parser.add_argument('--lbd', type=float, default=0.99)
    parser.add_argument('--adapt_steps', type=int, default=1)

    run_args = parser.parse_args()
    j = julia.Julia()
    j.using("NGSIM")
    args_filepath = "./args/params.npz"
    if os.path.isfile(args_filepath):
        args = hyperparams.load_args(args_filepath)
    else:
        raise ValueError("No such params file")

    if run_args.use_multiagent:
        args.env_multiagent = True
        args.remove_ngsim_vehicles = run_args.remove_ngsim_vehicles

    if run_args.debug:
        collect_fn = single_process_collect_trajectories
    else:
        collect_fn = parallel_collect_trajectories

    prev_lane_name = None
    data_base_dir = "./preprocessing/data"
    total_error = {"overall": [],
                   "curve": [],
                   "lane_change": [],
                   "straight": [],
                   "time_info": [],
                   "orig_traj": [],
                   "pred_traj": []}
    for dir_name in os.listdir(data_base_dir):
        if "downsampled" not in dir_name and os.path.isdir(os.path.join(data_base_dir, dir_name, "processed")):
            dir_error = {"overall": [],
                         "curve": [],
                         "lane_change": [],
                         "straight": [],
                         "time_info": [],
                         "orig_traj": [],
                         "pred_traj": []}
            for file_name in os.listdir(os.path.join(data_base_dir, dir_name, "processed")):
                try:
                    # if "section" in file_name:
                    #     orig_traj_file = os.path.join(dir_name, "processed", file_name)
                    #     print("processing file {}".format(orig_traj_file))
                    # else:
                    #     print("lane file, skipping")
                    #     continue
                    # lane_file = os.path.join(dir_name, "processed", '{}_lane'.format(orig_traj_file[:19]))
                    # processed_data_path = 'holo_{}_perfect_cleaned.csv'.format(orig_traj_file[5:19])
                    # df_len = clean_data(orig_traj_file)
                    # if df_len == 0:
                    #     print("Invalid file, skipping")
                    #     continue
                    # csv2txt(processed_data_path)
                    # if prev_lane_name != lane_file:
                    #     create_lane(lane_file)
                    # else:
                    #     print("Using same lane file, skipping generating a new one")
                    # print("Finish cleaning the original data")
                    # print("Start generating roadway")
                    # if prev_lane_name != lane_file:
                    #     base_dir = os.path.expanduser('~/Autoenv/data/')
                    #     j.write_roadways_to_dxf(base_dir)
                    #     j.write_roadways_from_dxf(base_dir)
                    # prev_lane_name = lane_file
                    # print("Finish generating roadway")
                    convert_raw_ngsim_to_trajdatas()
                    print("Start feature extraction")
                    extract_ngsim_features(output_filename="ngsim_holo_new.h5", n_expert_files=1)
                    print("Finish converting and feature extraction")

                    fn = "trajdata_holo_trajectories.txt"

                    hn = './data/trajectories/ngsim_holo_new.h5'

                    if run_args.n_envs:
                        args.n_envs = run_args.n_envs
                    # args.env_H should be 200
                    sys.stdout.write('{} vehicles with H = {}\n'.format(args.n_envs, args.env_H))

                    args.ngsim_filename = fn
                    args.h5_filename = hn
                    if args.env_multiagent:
                        egoids, _ = load_egoids(fn, args, run_args.n_runs_per_ego_id)
                    else:
                        egoids, _ = load_egoids(fn, args, run_args.n_runs_per_ego_id)
                    print("egoids")
                    print(egoids)
                    # print("starts")
                    # print(starts)

                    if len(egoids) == 0:
                        print("No valid vehicles, skipping")
                        continue
                    starts = None
                    error = collect(
                        egoids,
                        starts,
                        args,
                        exp_dir=run_args.exp_dir,
                        max_steps=200,
                        params_filename=run_args.params_filename,
                        use_hgail=run_args.use_hgail,
                        n_proc=run_args.n_proc,
                        collect_fn=collect_fn,
                        random_seed=run_args.random_seed,
                        lbd=run_args.lbd,
                        adapt_steps=run_args.adapt_steps
                    )
                    print("Vehicle Counter: {}".format(Veh_counter))
                except BaseException as e:
                    print("error occurred which is:{}".format(e))
                    continue
                exit(0)
                print("\n\nDirectory: {}, file: {} Statistical Info:\n\n".format(dir_name, file_name))
                utils.print_error(error)
                dir_error["overall"] += error["overall"]
                dir_error["curve"] += error["curve"]
                dir_error["lane_change"] += error["lane_change"]
                dir_error["straight"] += error["straight"]
                dir_error["time_info"] += error["time_info"]
                dir_error["orig_traj"] += error["orig_traj"]
                dir_error["pred_traj"] += error["pred_traj"]
            print("\n\nDirectory: {} Statistical Info:\n\n".format(dir_name))
            utils.print_error(dir_error)
            total_error["overall"] += dir_error["overall"]
            total_error["curve"] += dir_error["curve"]
            total_error["lane_change"] += dir_error["lane_change"]
            total_error["straight"] += dir_error["straight"]
            total_error["time_info"] += dir_error["time_info"]
            total_error["orig_traj"] += dir_error["orig_traj"]
            total_error["pred_traj"] += dir_error["pred_traj"]
            print("\n\nOverall Statistical Info up to now:\n\n")
            utils.print_error(total_error)

