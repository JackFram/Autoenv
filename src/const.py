import os
from src.Roadway import roadway
"""
    const value saving file
"""


DIR, filename = os.path.split(os.path.abspath(__file__))

FLOATING_POINT_REGEX = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
METERS_PER_FOOT = 0.3048

NGSIM_TRAJDATA_PATHS = [
                        os.path.join(DIR, "../data/i101_trajectories-0750am-0805am.txt"),
                        os.path.join(DIR, "../data/i101_trajectories-0805am-0820am.txt"),
                        os.path.join(DIR, "../data/i101_trajectories-0820am-0835am.txt"),
                        os.path.join(DIR, "../data/i80_trajectories-0400-0415.txt"),
                        os.path.join(DIR, "../data/i80_trajectories-0500-0515.txt"),
                        os.path.join(DIR, "../data/i80_trajectories-0515-0530.txt"),
                       ]

with open(os.path.join(DIR, "../data/ngsim_80.txt"), "r") as fp_80:
    ROADWAY_80 = roadway.read_roadway(fp_80)
    fp_80.close()

with open(os.path.join(DIR, "../data/ngsim_101.txt"), "r") as fp_101:
    ROADWAY_101 = roadway.read_roadway(fp_101)
    fp_101.close()

with open(os.path.join(DIR, "../data/ngsim_HOLO.txt"), "r") as fp_holo:
    ROADWAY_HOLO = roadway.read_roadway(fp_holo)
    fp_holo.close()

TRAJDATA_PATHS = [os.path.join( DIR, "../data/trajdata_i101_trajectories-0750am-0805am.txt"),
                os.path.join( DIR, "../data/trajdata_i101_trajectories-0805am-0820am.txt"),
                os.path.join( DIR, "../data/trajdata_i101_trajectories-0820am-0835am.txt"),
                os.path.join( DIR, "../data/trajdata_i80_trajectories-0400-0415.txt"),
                os.path.join( DIR, "../data/trajdata_i80_trajectories-0500-0515.txt"),
                os.path.join( DIR, "../data/trajdata_i80_trajectories-0515-0530.txt")]

NGSIM_TIMESTEP = 0.1 # [sec]
SMOOTHING_WIDTH_POS = 0.5 # [s]

'''
Const
'''
NGSIM_FILENAME_TO_ID = {
    'trajdata_i101_trajectories-0750am-0805am.txt': 1,
    'trajdata_i101_trajectories-0805am-0820am.txt': 2,
    'trajdata_i101_trajectories-0820am-0835am.txt': 3,
    'trajdata_i80_trajectories-0400-0415.txt': 4,
    'trajdata_i80_trajectories-0500-0515.txt': 5,
    'trajdata_i80_trajectories-0515-0530.txt': 6,
    'trajdata_i101-22agents-0750am-0805am.txt': 1,
    'trajdata_holo_trajectories.txt' : 1
}