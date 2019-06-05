from feature_extractor.feature_extractor import MultiFeatureExtractor
import os
from src.trajdata import load_trajdata, get_corresponding_roadway
from src.Record.frame import Frame
from src.Record.record import get_scene
import pickle, random


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


def index_ngsim_trajectory(filepath: str, minlength: int = 100, offset: int = 500, verbose: int = 1):

    # setup
    index = dict()
    trajdata = load_trajdata(filepath)
    n_frames = trajdata.nframes
    scene_length = max(trajdata.n_objects_in_frame(i) for i in range(trajdata.nframes))
    scene = Frame()
    scene.init(scene_length)
    prev, cur = set(), set()

    # iterate each frame collecting info about the vehicles
    for frame in range(offset - 1, n_frames - offset):
        if verbose > 0:
            print("\rframe {} / {}".format(frame, n_frames - offset))

        cur = set()
        scene = get_scene(scene, trajdata, frame)

        # add all the vehicles to the current set
        for veh in scene:
            cur.add(veh.id)
            # insert previously unseen vehicles into the index
            if veh.id not in prev:
                index[veh.id] = {"ts": frame}

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
    for (vehid, infos) in index:
        # check for start and end frames
        if "ts" not in infos.keys() or "te" not in infos.keys():
            if verbose > 0:
                print("delete vehid {} for missing keys".format(vehid))
            index.pop(vehid)
        elif infos["te"] - infos["ts"] < minlength:
            if verbose > 0:
                print("delete vehid {} for below minlength".format(vehid))
            index.pop(vehid)

    return index


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


def load_ngsim_trajdatas(filepaths, minlength: int=100):

    # check that indexes exist for the relevant trajdatas
    # if they are missing, create the index
    # the index is just a collection of metadata that is saved with the
    # trajdatas to allow for a more efficient environment implementation

    indexes_filepaths = [f.replace(".txt", "-index-{}.pkl".format(minlength)) for f in filepaths]
    indexes = []

    for (i, index_filepath) in enumerate(indexes_filepaths):
        if not os.path.isfile(index_filepath):
            index = index_ngsim_trajectory(filepaths[i], minlength=minlength)
            # TODO: finish save pickle file
            with open(index_filepath, "wb") as fp:
                pickle.dump(index, fp)
        else:
            with open(index_filepath, "rb") as fp:
                index = pickle.load(fp)

        indexes.append(index)

    trajdatas = []  # list of Records.ListRecord
    roadways = []  # list of Roadway

    for filepath in filepaths:
        trajdata = load_trajdata(filepath)
        trajdatas.append(trajdata)
        roadway = get_corresponding_roadway(filepath)
        roadways.append(roadway)

    return trajdatas, indexes, roadways





