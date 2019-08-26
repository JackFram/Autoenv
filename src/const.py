import os
from src.Roadway import roadway
"""
    const value saving file
"""


DIR, filename = os.path.split(os.path.abspath(__file__))

FLOATING_POINT_REGEX = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
METERS_PER_FOOT = 0.3048

NGSIM_TRAJDATA_PATHS = [os.path.join(DIR, "../data/holo_trajectories.txt"),
                        # os.path.join(DIR, "../data/i101_trajectories-0750am-0805am.txt")
                        ]

TRAJDATA_PATHS = [os.path.join(DIR, "../data/trajdata_holo_trajectories.txt")]

NGSIM_TIMESTEP = 0.1  # [sec]
SMOOTHING_WIDTH_POS = 0.5  # [s]

'''
Const
'''
NGSIM_FILENAME_TO_ID = {
    'trajdata_i101-22agents-0750am-0805am.txt': 0,
    'trajdata_holo_trajectories.txt': 0
}