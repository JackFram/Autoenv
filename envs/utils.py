from feature_extractor.feature_extractor import MultiFeatureExtractor
import os
import math
from src.trajdata import load_trajdata, get_corresponding_roadway
from src.Record.frame import Frame
from src.Record.record import get_scene
from src.const import NGSIM_FILENAME_TO_ID
import hgail.misc.utils
import rllab.spaces
import numpy as np
import random
import h5py


def build_space(shape, space_type, info={}):
    if space_type == 'Box':
        if 'low' in info and 'high' in info:
            low = np.array(info['low'])
            high = np.array(info['high'])
            msg = 'shape = {}\tlow.shape = {}\thigh.shape={}'.format(
                shape, low.shape, high.shape)
            assert shape == low.shape and shape == high.shape, msg
            return rllab.spaces.Box(low=low, high=high)
        else:
            return rllab.spaces.Box(low=-np.inf, high=np.inf, shape=shape)
    elif space_type == 'Discrete':
        assert len(shape) == 1, 'invalid shape for Discrete space'
        return rllab.spaces.Discrete(shape)
    else:
        raise(ValueError('space type not implemented: {}'.format(space_type)))


def dict_get(d: dict, key, default):
    if key in d:
        return d[key]
    else:
        return default


def max_n_objects(trajdatas: list):
    cur_max = -1
    for trajdata in trajdatas:
        cur = max(trajdata.n_objects_in_frame(i) for i in range(trajdata.nframes))
        cur_max = max(cur, cur_max)
    return cur_max


def fill_infos_cache(ext: MultiFeatureExtractor):
    cache = dict()
    cache["feature_names"] = ext.feature_names()

    for (i, n) in enumerate(cache["feature_names"]):
        if "is_colliding" == n:
            cache["is_colliding_idx"] = i
        if "out_of_lane" == n:
            cache["out_of_lane_idx"] = i
        if "markerdist_left" == n:
            cache["markerdist_left_idx"] = i
        if "markerdist_right" == n:
            cache["markerdist_right_idx"] = i
        if "accel" == n:
            cache["accel_idx"] = i
        if "distance_road_edge_right" == n:
            cache["distance_road_edge_right_idx"] = i
        if "distance_road_edge_left" == n:
            cache["distance_road_edge_left_idx"] = i

    return cache


def index_ngsim_trajectory(filepath: str, minlength: int = 100, offset: int = 0, verbose: int = 1):

    # setup
    index = dict()
    trajdata = load_trajdata(filepath)
    n_frames = trajdata.nframes
    print("n_frames: ", n_frames)
    scene_length = max(trajdata.n_objects_in_frame(i) for i in range(trajdata.nframes))
    scene = Frame()
    scene.init(scene_length)
    prev, cur = set(), set()

    # iterate each frame collecting info about the vehicles
    for frame in range(offset, n_frames - offset):
        # if verbose > 0:
        #     print("\rframe {} / {}".format(frame, n_frames - offset))

        cur = set()
        scene = get_scene(scene, trajdata, frame)

        # add all the vehicles to the current set
        # print("=======frame {}: vehicle num: {}==========\n".format(frame, scene.n))
        for i in range(scene.n):
            veh = scene[i]
            cur.add(veh.id)
            # print("vehicle id: {}".format(veh.id))
            # insert previously unseen vehicles into the index
            if veh.id not in prev:
                index[veh.id] = {"ts": frame}
                # print("vehicle {} start from {}".format(veh.id, frame))

        # find vehicles in the previous but not the current frame
        missing = prev - cur
        for id in missing:
            # set the final frame for all these vehicles
            index[id]["te"] = frame - 1

        # step forward
        prev = cur

    # at this point, any ids in cur are in the last frame, so add them in
    for id in cur:
        index[id]["te"] = n_frames - offset

    # postprocess to remove undesirable trajectories
    del_items = []
    for (vehid, infos) in index.items():
        # check for start and end frames
        if "ts" not in infos.keys() or "te" not in infos.keys():
            if verbose > 0:
                print("delete vehid {} for missing keys".format(vehid))
            del_items.append(vehid)
        elif infos["te"] - infos["ts"] < minlength:
            if verbose > 0:
                print("delete vehid {} for below minlength".format(vehid))
            del_items.append(vehid)

    for i in del_items:
        index.pop(i)

    return index


def random_sample_from_set_without_replacement(s: set, n):
    assert len(s) >= n
    sampled = set()
    for i in range(n):
        cur = random.choice(list(s))
        sampled.add(cur)
        s.discard(cur)
    return list(sampled)


def sample_trajdata_vehicle(trajinfos, offset: int = 0, traj_idx: int = None, egoid: int = None,
                            start: int = None):
    if traj_idx is None or egoid is None or start is None:
        traj_idx = random.randint(0, len(trajinfos) - 1)
        egoid = random.choice(list(trajinfos[traj_idx].keys()))
        ts = trajinfos[traj_idx][egoid]["ts"]
        te = trajinfos[traj_idx][egoid]["te"]
        ts = random.randint(ts, te - offset)
    else:
        ts = start
        te = start + offset

    return traj_idx, egoid, ts, te


def sample_multiple_trajdata_vehicle(n_veh: int, trajinfos, offset: int, max_resamples: int = 100, egoid: int = None,
                                     traj_idx: int = None, verbose: bool = True, rseed: int = None):
    if rseed is not None:
        random.seed(rseed)
    # if passed in egoid and traj_idx, use those, otherwise, sample
    if egoid is None or traj_idx is None:
        traj_idx = random.randint(0, len(trajinfos) - 1)
        egoid = random.choice(list(trajinfos[traj_idx].keys()))

    ts = trajinfos[traj_idx][egoid]["ts"]
    te = trajinfos[traj_idx][egoid]["te"]
    # this sampling assumes ts:te-offset is a valid range
    # this is enforced by the initial computation of the index / trajinfo
    ts = random.randint(ts, te - offset)
    # after setting the start timestep randomly from the valid range, next
    # update the end timestep to be offset timesteps following it
    # this assume that we just want to simulate for offset timesteps
    te = ts + offset
    # find all other vehicles that have at least 'offset' many steps in common
    # with the first sampled egoid starting from ts. If the number of such
    # vehicles is fewer than n_veh, then resample
    # start with the set containing the first egoid so we don't double count it
    egoids = set()
    egoids.add(egoid)
    for othid in trajinfos[traj_idx].keys():
        oth_ts = trajinfos[traj_idx][othid]["ts"]
        oth_te = trajinfos[traj_idx][othid]["te"]
        # other vehicle must start at or before ts and must end at or after te
        if oth_ts <= ts and te <= oth_te:
            egoids.add(othid)

    # check that there are enough valid ids from which to sample
    if len(egoids) < n_veh:
        # if not, resample
        # this is not ideal, but dramatically simplifies the multiagent env
        # if it becomes a problem, implement a version of the multiagent env
        # with asynchronous resets
        if verbose:
            print(
                "WARNING: insuffcient sampling ids in sample_multiple_trajdata_vehicle,\
                 resamples remaining: {}".format(max_resamples)
            )
        if max_resamples == 0:
            raise ValueError("ERROR: reached maximum resamples in sample_multiple_trajdata_vehicle")
        else:
            return sample_multiple_trajdata_vehicle(
                n_veh,
                trajinfos,
                offset,
                max_resamples=max_resamples - 1,
                verbose=verbose
            )

    # reaching this point means there are sufficient ids, sample the ones to use
    egoids = random_sample_from_set_without_replacement(egoids, n_veh)
    return traj_idx, egoids, ts, te


def select_multiple_trajdata_vehicle(n_veh: int, trajinfos, offset: int, max_resamples: int = 100, egoid: int = None,
                                     traj_idx: int = None, verbose: bool = True, period_start: int = 0, period_end: int = 100000,
                                     rseed: int = None, multiple: bool=False):
    assert traj_idx is not None
    assert egoid is not None
    if rseed is not None:
        random.seed(rseed)

    ts = max(trajinfos[traj_idx][egoid]["ts"], period_start)
    te = min(trajinfos[traj_idx][egoid]["te"], period_end)
    # find all other vehicles that have at least 'offset' many steps in common
    # with the first sampled egoid starting from ts. If the number of such
    # vehicles is fewer than n_veh, then resample
    # start with the set containing the first egoid so we don't double count it
    # print("id ts te")
    # print("egoid: ")
    # print("{} {} {}".format(egoid, ts, te))
    egoids = set([egoid])
    # print("egoids: ")
    if multiple:
        for othid in trajinfos[traj_idx].keys():
            oth_ts = trajinfos[traj_idx][othid]["ts"]
            oth_te = trajinfos[traj_idx][othid]["te"]
            # print("{} {} {}".format(othid, oth_ts, oth_te))
            ts_ = trajinfos[traj_idx][egoid]["ts"]
            te_ = trajinfos[traj_idx][egoid]["te"]
            # other vehicle must start at or before ts and must end at or after te
            if oth_ts <= ts_ and te_ <= oth_te:
                # print("pushing " + str(othid) + " into egoids")
                egoids.add(othid)
    n_veh = len(egoids)
    # print("n_veh: ", n_veh)
    # check that there are enough valid ids from which to select
    if len(egoids) < n_veh:
        if verbose:
            print(
                "WARNING: insuffcient sampling ids in sample_multiple_trajdata_vehicle, \
                resamples remaining: $(max_resamples)")

    # This should be based on relative distance or something.
    egoids = list(egoids)
    egoids = sorted(egoids)
    return traj_idx, egoids, ts, te


def load_ngsim_trajdatas(filepaths, minlength: int=100):

    # check that indexes exist for the relevant trajdatas
    # if they are missing, create the index
    # the index is just a collection of metadata that is saved with the
    # trajdatas to allow for a more efficient environment implementation

    indexes_filepaths = [f.replace(".txt", "-index-{}-ids.h5".format(minlength)) for f in filepaths]
    indexes = []

    for (i, index_filepath) in enumerate(indexes_filepaths):
        print(index_filepath)
        # if not os.path.isfile(index_filepath):
        if True:
            index = index_ngsim_trajectory(filepaths[i], minlength=minlength, verbose=0)
            ids = list(index.keys())
            # print("index ids: ", ids)
            ts = []
            te = []
            for id in ids:
                ts.append(index[id]["ts"])
                te.append(index[id]["te"])
            file = h5py.File(index_filepath, 'w')
            file.create_dataset('ids', data=ids)
            file.create_dataset('ts', data=ts)
            file.create_dataset('te', data=te)
            file.close()
        else:
            ids_file = h5py.File(index_filepath, 'r')
            index = {}
            ids = ids_file['ids'].value
            ts = ids_file['ts'].value
            te = ids_file['te'].value
            for j, id in enumerate(ids):
                index[id] = {"ts": ts[j], "te": te[j]}
        indexes.append(index)

    trajdatas = []  # list of Records.ListRecord
    roadways = []  # list of Roadway

    for filepath in filepaths:
        trajdata = load_trajdata(filepath)
        trajdatas.append(trajdata)
        roadway = get_corresponding_roadway(filepath)
        roadways.append(roadway)

    return trajdatas, indexes, roadways


def create_index_file(filepaths, minlength: int=100):
    indexes_filepaths = [f.replace(".txt", "-index-{}-ids.h5".format(minlength)) for f in filepaths]
    for (i, index_filepath) in enumerate(indexes_filepaths):
        print(index_filepath)
        # if not os.path.isfile(index_filepath):
        index = index_ngsim_trajectory(filepaths[i], minlength=minlength, verbose=0)
        ids = list(index.keys())
        print("index ids: ", ids)
        ts = []
        te = []
        for id in ids:
            ts.append(index[id]["ts"])
            te.append(index[id]["te"])
        file = h5py.File(index_filepath, 'w')
        file.create_dataset('ids', data=ids)
        file.create_dataset('ts', data=ts)
        file.create_dataset('te', data=te)
        file.close()


def load_x_feature_names(filepath, ngsim_filename):
    print("obs feature: h5 file path: {}, ngsim file name: {}".format(filepath, ngsim_filename))
    f = h5py.File(filepath, 'r')
    xs = []
    traj_id = NGSIM_FILENAME_TO_ID[ngsim_filename]
    # in case this nees to allow for multiple files in the future
    traj_ids = [traj_id]
    veh_2_index = dict()
    for i in traj_ids:
        feature_name = str(i) + "_feature"
        index_name = str(i) + "_index"
        if feature_name in f.keys():
            xs.append(f[feature_name])
        else:
            raise ValueError('invalid key to trajectory data: {}_feature'.format(i))
        if index_name in f.keys():
            veh_2_index[i] = f[index_name]
        else:
            raise ValueError('invalid key to trajectory index: {}_index'.format(i))
    x = np.concatenate(xs)
    feature_names = []
    for feature_name in f.attrs['feature_names']:
        feature_names.append(feature_name.decode())
    return x, veh_2_index, feature_names


def compute_lengths(arr):
    sums = np.sum(np.array(arr), axis=2)
    lengths = []
    for sample in sums:
        zero_idxs = np.where(sample == 0.)[0]
        if len(zero_idxs) == 0:
            lengths.append(len(sample))
        else:
            lengths.append(zero_idxs[0])
    return np.array(lengths)


def normalize(x, clip_std_multiple=np.inf):
    mean = np.mean(x, axis=0, keepdims=True)
    x = x - mean
    std = np.std(x, axis=0, keepdims=True) + 1e-8
    up = std * clip_std_multiple
    lb = - std * clip_std_multiple
    x = np.clip(x, lb, up)
    x = x / std
    return x, mean, std


def normalize_range(x, low, high):
    low = np.array(low)
    high = np.array(high)
    mean = (high + low) / 2.
    half_range = (high - low) / 2.
    x = (x - mean) / half_range
    x = np.clip(x, -1, 1)
    return x


def load_data(
        filepath,
        act_keys=['accel', 'turn_rate_global'],
        ngsim_filename='trajdata_i101_trajectories-0750am-0805am.txt',
        debug_size=None,
        min_length=50,
        normalize_data=True,
        shuffle=False,
        act_low=-1,
        act_high=1,
        clip_std_multiple=np.inf):
    # loading varies based on dataset type
    x, feature_names = load_x_feature_names(filepath, ngsim_filename)

    # optionally keep it to a reasonable size
    if debug_size is not None:
        x = x[:debug_size]

    if shuffle:
        idxs = np.random.permutation(len(x))
        x = x[idxs]

    # compute lengths of the samples before anything else b/c this is fragile
    lengths = compute_lengths(x)

    # flatten the dataset to (n_samples, n_features)
    # taking only the valid timesteps from each sample
    # i.e., throw out timeseries information
    xs = []
    for i, l in enumerate(lengths):
        # enforce minimum length constraint
        if l >= min_length:
            xs.append(x[i, :l])
    x = np.concatenate(xs)

    # split into observations and actions
    # redundant because the environment is not able to extract actions
    obs = x
    act_idxs = [i for (i, n) in enumerate(feature_names) if n in act_keys]
    act = x[:, act_idxs]

    if normalize_data:

        # normalize it all, _no_ test / val split
        obs, obs_mean, obs_std = normalize(obs, clip_std_multiple)
        # normalize actions to between -1 and 1
        act = normalize_range(act, act_low, act_high)

    else:
        obs_mean = None
        obs_std = None

    return dict(
        observations=obs,
        actions=act,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )


def keep_vehicle_subset(scene: Frame, ids: list):
    keep_ids = set(ids)
    scene_ids = []
    for i in range(scene.n):
        scene_ids.append(scene[i].id)
    scene_ids = set(scene_ids)
    remove_ids = scene_ids - keep_ids
    for id in remove_ids:
        scene.delete_by_id(id)
    return scene

'''
This is about as hacky as it gets, but I want to avoid editing the rllab 
source code as much as possible, so it will have to do for now.
Add a reset(self, kwargs**) function to the normalizing environment
https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
'''


def normalize_env_reset_with_kwargs(self, **kwargs):
    ret = self._wrapped_env.reset(**kwargs)
    if self._normalize_obs:
        return self._apply_normalize_obs(ret)
    else:
        return ret


def add_kwargs_to_reset(env):
    normalize_env = hgail.misc.utils.extract_normalizing_env(env)
    if normalize_env is not None:
        normalize_env.reset = normalize_env_reset_with_kwargs.__get__(normalize_env)

'''end of hack, back to our regularly scheduled programming'''


'''
Common 
'''


def maybe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)


def str2bool(v):
    if v.lower() == 'true':
        return True
    return False


def write_trajectories(filepath, trajs):
    np.savez(filepath, trajs=trajs)


def partition_list(lst, n):
    sublists = [[] for _ in range(n)]
    for i, v in enumerate(lst):
        sublists[i % n].append(v)
    return sublists


def cal_overall_rmse(error, verbose=False):
    totalStep = len(error)
    if totalStep == 0:
        return None
    if verbose:
        print("======================================")
        print("Calculating overall RMSE:\n")
        print("total step: ", totalStep)
    sum_dx = 0
    sum_dy = 0
    sum_dist = 0
    step = 0
    for i in range(len(error)):
        for j in range(len(error[i])):
            for k in range(len(error[i][j])):
                sum_dx += error[i][j][k]["dx"]
                sum_dy += error[i][j][k]["dy"]
                sum_dist += error[i][j][k]["dist"]
                step += 1
    if step == 0:
        return None
    avg_dx = sum_dx / step
    avg_dy = sum_dy / step
    avg_dist = sum_dist / step
    if verbose:
        print("avg dx: ", avg_dx)
        print("avg dy: ", avg_dy)
        print("avg dist: ", avg_dist)
    return {"dx": avg_dx, "dy": avg_dy, "dist": avg_dist}


def cal_step_rmse(error, i, verbose=False):
    totalStep = len(error)
    predict_span = len(error[0])
    n_agent = len(error[0][0])
    if verbose:
        print("Calculating step RMSE at step {}:".format(i))
        print("total step: ", totalStep)
        print("predict span: ", predict_span)
        print("n_agent: ", n_agent)
    sum_dx = 0
    sum_dy = 0
    sum_dist = 0
    for j in range(predict_span):
        for k in range(n_agent):
            sum_dx += error[i][j][k]["dx"]
            sum_dy += error[i][j][k]["dy"]
            sum_dist += error[i][j][k]["dist"]
    avg_dx = sum_dx / (predict_span * n_agent)
    avg_dy = sum_dy / (predict_span * n_agent)
    avg_dist = sum_dist / (predict_span * n_agent)
    if verbose:
        print("avg dx: ", avg_dx)
        print("avg dy: ", avg_dy)
        print("avg dist: ", avg_dist)
    return avg_dx, avg_dy, avg_dist


def cal_lookahead_rmse(error, j, verbose=False):
    totalStep = len(error)
    if totalStep == 0:
        return None, None, None
    predict_span = len(error[0])
    if verbose:
        print("======================================")
        print("Calculating look ahead RMSE at step {}:\n".format(j))
        print("total step: ", totalStep)
        print("predict span: ", predict_span)
    sum_dx = 0
    sum_dy = 0
    sum_dist = 0
    step = 0
    for i in range(totalStep):
        if j >= len(error[i]):
            break
        for k in range(1):
            sum_dx += error[i][j][k]["dx"]
            sum_dy += error[i][j][k]["dy"]
            sum_dist += error[i][j][k]["dist"]
            step += 1
    if step == 0:
        return None, None, None
    avg_dx = sum_dx / step
    avg_dy = sum_dy / step
    avg_dist = sum_dist / step
    if verbose:
        print("avg dx: ", avg_dx)
        print("avg dy: ", avg_dy)
        print("avg dist: ", avg_dist)
    return avg_dx, avg_dy, avg_dist


def cal_agent_rmse(error, k, verbose=False, lookahead_span=10):
    totalStep = len(error)
    predict_span = len(error[0])
    n_agent = len(error[0][0])
    if verbose:
        print("======================================")
        print("Calculating agent {} RMSE:\n".format(k))
        print("total step: ", totalStep)
        print("predict span: ", predict_span)
        print("n_agent: ", n_agent)
    sum_dx = 0
    sum_dy = 0
    sum_dist = 0
    step = 0
    for i in range(len(error)):
        for j in range(len(error[i])):
            sum_dx += error[i][j][k]["dx"]
            sum_dy += error[i][j][k]["dy"]
            sum_dist += error[i][j][k]["dist"]
            step += 1
    avg_dx = sum_dx / step
    avg_dy = sum_dy / step
    avg_dist = sum_dist / step
    if verbose:
        print("avg dx: ", avg_dx)
        print("avg dy: ", avg_dy)
        print("avg dist: ", avg_dist)
    step_error_dx = []
    step_error_dy = []
    step_error_dist = []
    for j in range(lookahead_span):
        per_step = 0
        per_sum_dx = 0
        per_sum_dy = 0
        per_sum_dist = 0
        for i in range(len(error)):
            if j >= len(error[i]):
                break
            per_sum_dx += error[i][j][k]["dx"]
            per_sum_dy += error[i][j][k]["dy"]
            per_sum_dist += error[i][j][k]["dist"]
            per_step += 1
        step_error_dx.append(per_sum_dx / per_step)
        step_error_dy.append(per_sum_dy / per_step)
        step_error_dist.append(per_sum_dist / per_step)
    print("look ahead avg dx: ", step_error_dx)
    print("look ahead avg dy: ", step_error_dy)
    print("look ahead avg dist: ", step_error_dist)
    return {"dx": avg_dx, "dy": avg_dy, "dist": avg_dist}


def cal_m_stability(error, h=50, T=150, e=0.1, verbose=False):
    # error (step, predicted_span, n_agent)
    n_agent = len(error[0][0])
    error_time = []

    for k in range(h, T):
        step = h
        for j in range(0, h):
            flag = 0
            for i in range(n_agent):
                if error[k-j][j][i]["dist"] > error[k-j-1][j+1][i]["dist"] and error[k-j][j][i]["dist"] > e:
                    flag = 1

            if flag:
                step = j
                break
        error_time.append(step)
    return error_time


def cal_lookahead(error: list, predict_span: int):
    if len(error) == 0:
        return None
    rmse_over_lookahead_dx = []
    rmse_over_lookahead_dy = []
    rmse_over_lookahead_dist = []
    for j in range(predict_span):  # this should be the span you want to test
        dx, dy, dist = cal_lookahead_rmse(error, j)
        rmse_over_lookahead_dx.append(dx)
        rmse_over_lookahead_dy.append(dy)
        rmse_over_lookahead_dist.append(dist)
    if len(rmse_over_lookahead_dist) == 0:
        return None
    return {"dx": rmse_over_lookahead_dx, "dy": rmse_over_lookahead_dy, "dist": rmse_over_lookahead_dist}


def cal_avg(error: list, type: str = None):
    assert type is not None
    n = len(error)
    dx = 0
    dy = 0
    dist = 0
    dx_std = 0
    dy_std = 0
    dist_std = 0
    lookahead_dx = [0 for _ in range(50)]
    lookahead_dy = [0 for _ in range(50)]
    lookahead_dist = [0 for _ in range(50)]
    lookahead_dx_std = [0 for _ in range(50)]
    lookahead_dy_std = [0 for _ in range(50)]
    lookahead_dist_std = [0 for _ in range(50)]
    step = 0
    lh_step = [0 for _ in range(50)]

    # calculate mean
    for i in range(n):
        if "lookahead" not in type and error[i][type] is not None:
            dx += error[i][type]["dx"]
            dy += error[i][type]["dy"]
            dist += error[i][type]["dist"]
            step += 1
        elif "lookahead" in type and error[i][type] is not None:
            lookahead_step = len(error[i][type]["dx"])
            # print(i, type, lookahead_step)
            for j in range(lookahead_step):
                if error[i][type]["dist"][j] is not None:
                    lh_step[j] += 1
                    lookahead_dx[j] += error[i][type]["dx"][j]
                    lookahead_dy[j] += error[i][type]["dy"][j]
                    lookahead_dist[j] += error[i][type]["dist"][j]
                else:
                    continue

    # calculate std
    for i in range(n):
        if "lookahead" not in type and error[i][type] is not None and step != 0:
            dx_std += (error[i][type]["dx"] - dx/step)**2
            dy_std += (error[i][type]["dy"] - dy/step)**2
            dist_std += (error[i][type]["dist"] - dist/step)**2
        elif "lookahead" in type and error[i][type] is not None:
            lookahead_step = len(error[i][type]["dx"])
            # print(i, type, lookahead_step)
            for j in range(lookahead_step):
                if error[i][type]["dist"][j] is not None and lh_step[j] != 0:
                    lookahead_dx_std[j] += (error[i][type]["dx"][j] - lookahead_dx[j]/lh_step[j])**2
                    lookahead_dy_std[j] += (error[i][type]["dy"][j] - lookahead_dy[j]/lh_step[j])**2
                    lookahead_dist_std[j] += (error[i][type]["dist"][j] - lookahead_dist[j]/lh_step[j])**2
                else:
                    continue
    if "lookahead" in type:
        for j in range(50):
            if lh_step[j] == 0:
                continue
            lookahead_dx[j] = lookahead_dx[j] / lh_step[j]
            lookahead_dy[j] = lookahead_dy[j] / lh_step[j]
            lookahead_dist[j] = lookahead_dist[j] / lh_step[j]
            lookahead_dx_std[j] = math.sqrt(lookahead_dx_std[j] / lh_step[j])
            lookahead_dy_std[j] = math.sqrt(lookahead_dy_std[j] / lh_step[j])
            lookahead_dist_std[j] = math.sqrt(lookahead_dist_std[j] / lh_step[j])
        return {"dx": lookahead_dx, "dy": lookahead_dy, "dist": lookahead_dist, "dx_std": lookahead_dx_std,
                "dy_std": lookahead_dy_std, "dist_std": lookahead_dist_std}
    if step == 0:
        return {"dx": 0, "dy": 0, "dist": 0, "dx_std": 0, "dy_std": 0, "dist_std": 0}
    else:
        if "lookahead" not in type:

            return {"dx": dx / step, "dy": dy / step, "dist": dist / step,
                    "dx_std": math.sqrt(dx_std / step), "dy_std": math.sqrt(dy_std / step),
                    "dist_std": math.sqrt(dist_std / step)}


def save_error(error: list):
    overall_rmse = cal_avg(error, "overall_rmse")
    curve_rmse = cal_avg(error, "curve_rmse")
    lane_change_rmse = cal_avg(error, "lane_change_rmse")
    straight_rmse = cal_avg(error, "straight_rmse")
    lh_overall_rmse = cal_avg(error, "overall_lookahead_rmse")
    lh_curve_rmse = cal_avg(error, "curve_lookahead_rmse")
    lh_laneChange_rmse = cal_avg(error, "lane_change_lookahead_rmse")
    lh_straight_rmse = cal_avg(error, "straight_lookahead_rmse")
    print("==========================================================")
    print("Overall RMSE:")
    print("dx: {} +- {}   dy: {} +- {}   dist: {} +- {}  [m]".format(overall_rmse['dx'], overall_rmse['dx_std'],
                                                            overall_rmse['dy'], overall_rmse['dy_std'],
                                                            overall_rmse['dist'], overall_rmse['dist_std']))
    print("Curve RMSE:")
    print("dx: {} +- {}   dy: {} +- {}   dist: {} +- {}  [m]".format(curve_rmse['dx'], curve_rmse['dx_std'],
                                                            curve_rmse['dy'], curve_rmse['dy_std'],
                                                            curve_rmse['dist'], curve_rmse['dist_std']))
    print("Lane Change RMSE:")
    print("dx: {} +- {}   dy: {} +- {}   dist: {} +- {}  [m]".format(lane_change_rmse['dx'], lane_change_rmse['dx_std'],
                                                            lane_change_rmse['dy'], lane_change_rmse['dy_std'],
                                                            lane_change_rmse['dist'], lane_change_rmse['dist_std']))
    print("Straight RMSE:")
    print("dx: {} +- {}   dy: {} +- {}   dist: {} +- {}  [m]".format(straight_rmse['dx'], straight_rmse['dx_std'],
                                                            straight_rmse['dy'], straight_rmse['dy_std'],
                                                            straight_rmse['dist'], straight_rmse['dist_std']))
    print("==========================================================")
    print("Overall RMSE:")
    print("dist")
    print(lh_overall_rmse["dist"])
    print("dist std")
    print(lh_overall_rmse["dist_std"])
    print("Curve RMSE:")
    print("dist")
    print(lh_curve_rmse["dist"])
    print("dist std")
    print(lh_curve_rmse["dist_std"])
    print("Lane Change RMSE:")
    print("dist")
    print(lh_laneChange_rmse["dist"])
    print("dist std")
    print(lh_laneChange_rmse["dist_std"])
    print("Straight RMSE:")
    print("dist")
    print(lh_straight_rmse["dist"])
    print("dist std")
    print(lh_straight_rmse["dist_std"])
    print("==========================================================")


def print_error(error: dict):
    overall_error = np.array(error["overall"])
    curve_error = np.array(error["curve"])
    lane_change_error = np.array(error["lane_change"])
    straight_error = np.array(error["straight"])
    time_list = np.array(error["time_info"])
    print("========================================================================================================")
    print("overall lane number: {}, curve: {}, lane_change: {}, straight: {}".format(overall_error.shape[0],
                                                                                     curve_error.shape[0],
                                                                                     lane_change_error.shape[0],
                                                                                     straight_error.shape[0]))
    print("50 steps, 0.1s/step:")
    if len(time_list) != 0:
        print("Average trajectory prediction time: {} seconds".format(time_list.mean(axis=0)[-1]))
    print("Look head RMSE:")
    print("Overall: ", overall_error.mean(axis=0))
    print("Look head RMSE std:")
    print("Overall: ", overall_error.std(axis=0))
    print("Range of RMSE per step:")
    if len(overall_error) != 0:
        print("min: {}\nmax: {}".format(overall_error.min(axis=0), overall_error.max(axis=0)))
        print("RMSE of Last predicted step: {}".format(overall_error.mean(axis=0)[-1]))
        print("Std of Last predicted step: {}".format(overall_error.std(axis=0)[-1]))
        print("Range of last step RMSE: ")
        print("min: {}, max: {}\n\n".format(overall_error.min(axis=0)[-1], overall_error.max(axis=0)[-1]))
        print("20 steps, 0.1s/step:")
        if len(time_list) != 0:
            print("Average trajectory prediction time: {} seconds".format(time_list.mean(axis=0)[0]))
        overall_error = overall_error[:, :20]
        print("Overall: ", overall_error.mean(axis=0))
        print("Look head RMSE std:")
        print("Overall: ", overall_error.std(axis=0))
        print("Range of RMSE per step:")
        print("min: {}\nmax: {}".format(overall_error.min(axis=0), overall_error.max(axis=0)))
        print("RMSE of Last predicted step: {}".format(overall_error.mean(axis=0)[-1]))
        print("Std of Last predicted step: {}".format(overall_error.std(axis=0)[-1]))
        print("Range of last step RMSE: ")
        print("min: {}, max: {}".format(overall_error.min(axis=0)[-1], overall_error.max(axis=0)[-1]))
    print("========================================================================================================\n")









