from feature_extractor.feature_extractor import MultiFeatureExtractor
import os
from src.trajdata import load_trajdata, get_corresponding_roadway


def dict_get(d: dict, key, default):
    if key in d:
        return d[key]
    else:
        return default


def max_n_objects(trajdatas: list):
    cur_max = -1
    for trajdata in trajdatas:
        cur = max(trajdata.n_objects_in_frame(i) for i in range(trajdata.nframse))
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


def load_ngsim_trajdatas(filepaths, minlength: int=100):

    # check that indexes exist for the relevant trajdatas
    # if they are missing, create the index
    # the index is just a collection of metadata that is saved with the
    # trajdatas to allow for a more efficient environment implementation

    indexes_filepaths = [f.replace(".txt", "-index-{}.h5".format(minlength)) for f in filepaths]
    indexes = []

    for (i, index_filepath) in enumerate(indexes_filepaths):
        if not os.path.isfile(index_filepath):
            index = index_ngsim_trajectory(filepaths[i], minlength=minlength)
            # TODO: finish save h5 file

    trajdatas = []  # list of Records.ListRecord
    roadways = []  # list of Roadway

    for filepath in filepaths:
        trajdata = load_trajdata(filepath)
        trajdatas.append(trajdata)
        roadway = get_corresponding_roadway(filepath)
        roadways.append(roadway)

    return trajdatas, indexes, roadways





