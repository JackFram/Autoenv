import os
import numpy as np
import h5py

from feature_extractor.utils import build_feature_extractor
from src import const
from src.trajdata import load_trajdata, get_corresponding_roadway
from envs.utils import max_n_objects
from src.Record.frame import Frame
from src.Record.record import SceneRecord, get_scene


def extract_features(ext, trajdata, roadway, timestep_delta, record_length, offset, prime, maxframes):
    n_features = len(ext)
    scene_length = max_n_objects([trajdata])
    scene = Frame()
    scene.init(scene_length)
    rec = SceneRecord()
    rec.init(record_length, 0.1, scene_length)
    features = dict()
    ctr = 0
    n_frames = trajdata.nframes
    for frame in range(offset - prime - 1, offset - 1):
        scene = get_scene(scene, trajdata, frame)
        rec.update(scene)
    veh_features = ext.pull_features(rec, roadway, 1)

    print("offset")
    print(offset)
    print("n_frames-offset")
    print(n_frames - offset)

    veh_list = dict()

    for frame in range(offset - 1, (n_frames - offset)):
        scene = get_scene(scene, trajdata, frame)
        for (vidx, veh) in enumerate(scene):
            veh_list[veh.id] = 0

    print(list(veh_list.keys()))

    for frame in range(offset - 1, (n_frames - offset)):
        ctr += 1
        if maxframes is not None and ctr >= maxframes:
            break
        print("\rframe {} / {}\n".format(frame, (n_frames - offset)))

        # update the rec
        scene = get_scene(scene, trajdata, frame)
        rec.update(scene)

        # every timestep_delta step, extract features
        if frame % timestep_delta == 0:
            for (vidx, veh) in enumerate(scene):
                # extract features
                veh_features = ext.pull_features(rec, roadway, vidx)
                if veh.id not in features.keys():
                    features[veh.id] = np.zeros((n_features, 0))
                features[veh.id] = np.concatenate((features[veh.id], veh_features.reshape(n_features, 1)), axis=1)
            for veh_id in veh_list.keys():
                if scene.findfirst(veh_id) is None:
                    features[veh_id] = np.zeros((n_features, 0))
                features[veh_id] = np.concatenate((features[veh_id], np.zeros((n_features, 1))), axis=1)

    return features


def write_features(features, output_filepath, ext):
    n_features = len(ext)
    # compute max length across samples
    maxlen = 0
    for (traj_idx, feature_dict) in features:
        for (veh_id, veh_features) in feature_dict:
            maxlen = max(maxlen, veh_features.shape[1])
    print("max length across samples: {}".format(maxlen))
    # write trajectory features
    h5file = h5py.File(output_filepath, "w")
    for (traj_idx, feature_dict) in features:
        feature_array = np.zeros((n_features, maxlen, len(feature_dict)))
        for (idx, (veh_id, veh_features)) in enumerate(feature_dict):
            print("idx: {} veh_id: {}".format(idx, veh_id))
            feature_array[:, 0:veh_features.shape[1], idx] = veh_features.reshape(n_features, veh_features.shape[1], 1)
        h5file["{}".format(traj_idx)] = feature_array
    # write feature names
    h5file.attrs["feature_names"] = ext.feature_names()
    h5file.close()


def extract_ngsim_features(timestep_delta=1, record_length=10, offset=50, prime=10, maxframes=None,
                           output_filename="ngsim.h5", n_expert_files=1):
    '''
    :param timestep_delta: timesteps between feature extractions
    :param record_length: number of frames for record to track in the past
    :param offset: from ends of the trajectories TODO: offset was 500, holo data is too short
    :param prime:
    :param maxframes: nothing for no max
    :param output_filename:
    :param n_expert_files: number of time periods for which to extract.
    :return: no return, write features to output file
    '''

    ext = build_feature_extractor()
    features = dict()

    # extract
    for traj_idx in range(n_expert_files):
        data_name = const.TRAJDATA_PATHS[traj_idx]
        trajdata = load_trajdata(data_name)
        scene_length = max_n_objects([trajdata])
        scene = Frame()
        scene.init(scene_length)
        roadway = get_corresponding_roadway(data_name)
        features[traj_idx] = extract_features(
            ext,
            trajdata,
            roadway,
            timestep_delta,
            record_length,
            offset,
            prime,
            maxframes
        )
    output_filepath = os.path.join("../data/trajectories/", output_filename)
    print("output filepath: {}".format(output_filepath))
    write_features(features, output_filepath, ext)
