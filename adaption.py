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

from envs import hyperparams, utils, build_env

from envs.utils import str2bool
from algorithms.AGen import rls, validate_utils
from preprocessing.clean_holo import clean_data, csv2txt, create_lane
from preprocessing.extract_feature import extract_ngsim_features
from src.trajdata import convert_raw_ngsim_to_trajdatas
# import pdb
import math
import tqdm


plt.style.use("ggplot")

# TODO: change this accordingly
EGO_START_FRAME = 1106
N_VEH = 1
EGO_ID = 1978
DATA_INDEX = [96]
N_ITERATION = 1
MAX_STEP = 150


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
    start_time = time.time()
    theta = np.load('./data/theta.npy')  # TODO: change the file path
    theta = np.mean(theta)

    ego_start_frame = trajinfos[env_kwargs['egoid']]['ts']
    maxstep = trajinfos[env_kwargs['egoid']]['te'] - trajinfos[env_kwargs['egoid']]['ts'] - 50

    env_kwargs['start'] = ego_start_frame
    x = env.reset(**env_kwargs)

    n_agents = x.shape[0]
    print("Agent number: {}".format(n_agents))
    dones = [True] * n_agents
    predicted_trajs, adapnets = [], []
    policy.reset(dones)
    prev_actions, prev_hiddens = None, None

    # max_steps = min(200, obs.shape[1] - primesteps - 2)
    print("max_steps")
    print(maxstep)
    mean = np.expand_dims(mean, axis=2)
    prev_hiddens = np.zeros([n_agents, 64])

    param_length = 65 if adapt_steps == 1 else 195

    for i in range(n_agents):
        adapnets.append(rls.rls(lbd, theta, param_length, 2))

    avg = 0
    end_time = time.time()
    print(('Reset env Running time: %s Seconds' % (end_time - start_time)))
    lx = x
    error = []  # size is (maxstep, predict_span, n_agent) each element is a dict(dx: , dy: ,dist: )
    action_time = 0
    step_time = 0
    adaption_time = 0
    predict_time = 0
    reset_time = 0
    for step in tqdm.tqdm(range(ego_start_frame - 1, maxstep + ego_start_frame - 1)):

        # print("step = ", step)
        # print("feature: ", x)

        start = time.time()
        start_time = time.time()
        a, a_info, hidden_vec = policy.get_actions_with_prev(obs[:, step, :], mean[:, step, :], prev_hiddens)

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
        end_time = time.time()
        adaption_time += end_time - start_time
        prev_actions, prev_hiddens = a, hidden_vec
        start_time = time.time()
        traj, error_per_step, time_info = prediction(env_kwargs, obs[:, step + 1, :], adapnets, env, policy, prev_hiddens, n_agents, adapt_steps, nids)
        end_time = time.time()
        predict_time += (end_time - start_time)
        action_time += time_info["action"]
        step_time += time_info["step"]
        error.append(error_per_step)
        predicted_trajs.append(traj)
        d = np.stack([adapnets[i].draw for i in range(n_agents)])
        end = time.time()
        avg += (start - end)
        env_kwargs['start'] += 1
        lx = x
        start_time = time.time()
        x = env.reset(**env_kwargs)
        end_time = time.time()
        reset_time += end_time - start_time
        # print(('Step %d reset env Running time: %s Seconds' % (step, end_time - start_time)))
    # print(('predict trajectory time: %s Seconds' % (predict_time / maxstep)))
    # print(('env reset time: %s Seconds' % (reset_time / maxstep)))
    # print(('adaption time: %s Seconds' % (adaption_time / maxstep)))
    # print(('action time: %s Seconds' % (action_time / maxstep)))
    # print(('env step time: %s Seconds' % (step_time / maxstep)))

    # test
    # m_stability = utils.cal_m_stability(error, T=150)
    # with open("./m_stability.pkl", "wb") as fp:
    #     pickle.dump(m_stability, fp)
    #     print("finish saving M stability matrix.")
    error_info = dict()
    error_info["overall_rmse"] = utils.cal_overall_rmse(error, verbose=True)
    rmse_over_lookahead_dx = []
    rmse_over_lookahead_dy = []
    rmse_over_lookahead_dist = []
    for j in range(50):  # this should be the span you want to test
        dx, dy, dist = utils.cal_lookahead_rmse(error, j)
        rmse_over_lookahead_dx.append(dx)
        rmse_over_lookahead_dy.append(dy)
        rmse_over_lookahead_dist.append(dist)
    print("======================================")
    print("RMSE over look ahead score:\n")
    print("rmse over look ahead dx:")
    print(rmse_over_lookahead_dx[:10])
    print("rmse over look ahead dy:")
    print(rmse_over_lookahead_dy[:10])
    print("rmse over look ahead dist:")
    print(rmse_over_lookahead_dist[:10])
    error_info["lookahead_rmse"] = {"dx": rmse_over_lookahead_dx, "dy": rmse_over_lookahead_dy,
                                    "dist": rmse_over_lookahead_dist}

    error_info["agent_rmse"] = []
    for i in range(n_agents):
        error_info["agent_rmse"].append(utils.cal_agent_rmse(error, i, verbose=True))

    print('obs.shape')
    print(obs.shape)

    return predicted_trajs, error_info


def prediction(env_kwargs, x, adapnets, env, policy, prev_hiddens, n_agents, adapt_steps, nids):
    traj = hgail.misc.simulation.Trajectory()
    predict_span = 50
    error_per_step = []  # size is (predict_span, n_agent) each element is a dict(dx: , dy: ,dist: )
    get_action_time = 0
    env_step_time = 0
    for j in range(predict_span):
        # if j == 0:
        #     print("feature {}".format(j), x)
        start_time = time.time()
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

        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_std) + means
        end_time = time.time()
        get_action_time += (end_time - start_time)
        start_time = time.time()
        nx, r, dones, e_info = env.step(actions)
        traj.add(x, actions, r, a_info, e_info)
        end_time = time.time()
        env_step_time += (end_time - start_time)
        error_per_agent = []  # length is n_agent, each element is a dict(dx: , dy: ,dist: )

        for i in range(n_agents):
            # print("orig x: ", e_info["orig_x"][i])
            # print("orig y: ", e_info["orig_y"][i])
            # print("predicted x: ", e_info["x"][i])
            # print("predicted y: ", e_info["y"][i])
            dx = abs(e_info["orig_x"][i] - e_info["x"][i])
            dy = abs(e_info["orig_y"][i] - e_info["y"][i])
            dist = math.hypot(dx, dy)
            # print("{}-----> dx: {} dy: {} dist: {}".format(j, dx, dy, dist))
            error_per_agent.append({"dx": dx, "dy": dy, "dist": dist})
        error_per_step.append(error_per_agent)
        if any(dones): break
        x = nx

    time_info = {"action": (get_action_time / predict_span), "step": (env_step_time / predict_span)}
    # this should be delete and replaced
    # y = env.reset(**env_kwargs)

    return traj.flatten(), error_per_step, time_info


def collect_trajectories(
        args,
        params,
        egoids,
        starts,
        trajlist,
        pid,
        env_fn,
        policy_fn,
        max_steps,
        use_hgail,
        random_seed,
        lbd,
        adapt_steps):
    print('args')
    print(args)
    start_time = time.time()
    env, trajinfos, _, _ = env_fn(args, n_veh=N_VEH, alpha=0.)
    print(trajinfos[0])

    policy = policy_fn(args, env)
    end_time = time.time()
    print(('Initializing env Running time: %s Seconds' % (end_time - start_time)))
    with tf.Session() as sess:
        # initialize variables
        start_time = time.time()
        sess.run(tf.global_variables_initializer())

        # then load parameters
        if use_hgail:
            for i, level in enumerate(policy):
                level.algo.policy.set_param_values(params[i]['policy'])
            policy = policy[0].algo.policy
        else:
            policy.set_param_values(params['policy'])

        normalized_env = hgail.misc.utils.extract_normalizing_env(env)
        if normalized_env is not None:
            normalized_env._obs_mean = params['normalzing']['obs_mean']
            normalized_env._obs_var = params['normalzing']['obs_var']

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
        end_time = time.time()
        print(('Loading obs data Running time: %s Seconds' % (end_time - start_time)))
        if args.env_multiagent:
            # I add not because single simulation has no orig_x etc.
            # egoid = random.choice(egoids)
            trajinfos = trajinfos[0]
            error = []
            for veh_id in trajinfos.keys():
                if trajinfos[veh_id]["te"] - trajinfos[veh_id]["ts"] <= 50:
                    break
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

                trajlist.append(traj)
                error.append(error_info)
            utils.print_error(error)
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
                trajlist.append(traj)

        print('finish online adaption')

    return trajlist


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
    start_time = time.time()
    manager = mp.Manager()
    trajlist = manager.list()

    # set policy function
    policy_fn = build_env.build_hierarchy if use_hgail else validate_utils.build_policy

    # partition egoids
    proc_egoids = utils.partition_list(egoids, n_proc)

    # pool of processes, each with a set of ego ids
    pool = mp.Pool(processes=n_proc)
    end_time = time.time()
    print(('Creating parallel env Running time: %s Seconds' % (end_time - start_time)))
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
                trajlist,
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

    # wait for the processes to finish
    [res.get() for res in results]
    pool.close()
    # let the julia processes finish up
    time.sleep(10)
    return trajlist


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
    params_filepath = os.path.join(exp_dir, 'imitate/log/{}'.format(params_filename))
    params = hgail.misc.utils.load_params(params_filepath)
    # validation setup
    validation_dir = os.path.join(exp_dir, 'imitate', 'test')
    utils.maybe_mkdir(validation_dir)
    output_filepath = os.path.join(validation_dir, '{}_AGen_{}_{}.npz'.format(
        args.ngsim_filename.split('.')[0], adapt_steps, args.env_multiagent))

    with Timer():
        trajs = collect_fn(
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

    utils.write_trajectories(output_filepath, trajs)


def load_egoids(filename, args, n_runs_per_ego_id=10, env_fn=build_env.build_ngsim_env):
    offset = args.env_H + args.env_primesteps
    basedir = os.path.expanduser('~/Autoenv/data/')  # TODO: change the file path
    ids_filename = filename.replace('.txt', '-index-{}-ids.h5'.format(offset))
    print("ids_filename")
    print(ids_filename)
    ids_filepath = os.path.join(basedir, ids_filename)
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
        te = ids_file['te'].value - offset
        starts = np.array([np.random.randint(s, e + 1) for (s, e) in zip(ts, te)])
        # write to file
        starts_file = h5py.File(start_times_filepath, 'w')
        starts_file.create_dataset('starts', data=starts)
        starts_file.close()

    # create a dict from id to start time
    id2starts = dict()
    for (egoid, start) in zip(ids, starts):
        id2starts[egoid] = start

    ids = np.tile(ids, n_runs_per_ego_id)
    return ids, id2starts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validation settings')
    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--exp_dir', type=str, default='./data/experiments/multiagent_curr')
    parser.add_argument('--params_filename', type=str, default='itr_2000.npz')
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

    orig_traj_file = "2018-11-16-15-54-47_corrected_smoothed_section_6_9.csv"  # TODO: you can change this filename

    lane_file = '{}_lane'.format(orig_traj_file[:19])
    processed_data_path = 'holo_{}_perfect_cleaned.csv'.format(orig_traj_file[5:19])
    clean_data(orig_traj_file)
    csv2txt(processed_data_path)
    create_lane(lane_file)
    print("Finish cleaning the original data")
    print("Start generating roadway")
    base_dir = os.path.expanduser('~/Autoenv/data/')
    j.write_roadways_to_dxf(base_dir)
    j.write_roadways_from_dxf(base_dir)
    print("Finish generating roadway")
    convert_raw_ngsim_to_trajdatas()
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
        # args.n_envs gives the number of simultaneous vehicles
        # so run_args.n_multiagent_trajs / args.n_envs gives the number
        # of simulations to run overall
        # egoids = list(range(int(run_args.n_multiagent_trajs / args.n_envs)))
        #  starts = dict()
        egoids, starts = load_egoids(fn, args, run_args.n_runs_per_ego_id)
    else:
        egoids, starts = load_egoids(fn, args, run_args.n_runs_per_ego_id)

    print("egoids")
    print(egoids)
    print("starts")
    print(starts)

    if len(egoids) == 0:
        print("No valid vehicles, exit")
        exit(0)

    collect(
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