import os

from feature_extractor.utils import build_feature_extractor
from src import const
from src.trajdata import load_trajdata, get_corresponding_roadway
from envs.utils import max_n_objects
from src.Record.frame import Frame
from src.Record.record import SceneRecord


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
