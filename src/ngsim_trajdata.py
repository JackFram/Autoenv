# Zhihao Zhang
# NGSIM dataset processor dataloader

import pandas as pd
import os
from tqdm import tqdm


class NGSIMTrajdata:
    """
    NGSIMTrajdata
    The trajectory data stored in the original NGSIM dataset format.
    The dataset is a white-space separated text file with columns:
        id                  - I - Vehicle identification number (ascending by time of entry into section)
        frame               - I - Frame Identification number (ascending by start time), units 1/10 of a second
        n_frames_in_dataset - I - Total number of frames in which the vehicle appears in this data set, units 1/10 of a second
        epoch               - I - Elapsed time since Jan 1, 1970, in milliseconds
        local_x             - F - Lateral (X) coordinate of the front center of the vehicle with respect to the left-most edge of the section in the direction of travel, in feet
        local_y             - F - Longitudinal (Y) coordinate of the front center of the vehicle with respect to the entry edge of the section in the direction of travel, in feet
        global_x            - F - X Coordinate of the front center of the vehicle based on CA State Plane III in NAD83
        global_y            - F - Y Coordinate of the front center of the vehicle based on CA State Plane III in NAD83
        length              - F - Length of the vehicle, in feet
        width               - F - Width of the vehicle, in feet
        class               - I - vehicle class, 1 - motorcycle, 2 - auto, 3 - truck
        speed               - F - Instantaneous velocity of vehicle, in ft/second
        acc                 - F - Instantaneous acceleration of vehicle, in ft/second^2
        lane                - I - Current lane position of vehicle
        carind_front        - I - Vehicle Id of the lead vehicle in the same lane. A value of '0' represents no preceding vehicle
        carind_rear         - I - Vehicle Id of the vehicle following the subject vehicle in the same lane. A value of '0' represents no following vehicle
        dist_headway        - F - Spacing provides the distance between the front-center of a vehicle to the front-center of the preceding vehicle, in feet
        time_headway        - F - Headway provides the time to travel from the front-center of a vehicle (at the speed of the vehicle) to the front-center of the preceding vehicle. A headway value of 9999.99 means that the vehicle is traveling at zero speed (congested conditions), in second
    """
    def __init__(self, file_path: str):
        assert os.path.isfile(file_path)
        self.df = pd.read_csv(file_path, sep=" ", header=None, skipinitialspace=True)
        self.car2start = {}
        self.frame2cars = {}
        col_names = ['id', 'frame', 'n_frames_in_dataset', 'epoch', 'local_x',
                     'local_y', 'global_x', 'global_y', 'length', 'width', 'class',
                     'speed', 'acc', 'lane', 'carind_front', 'carind_rear',
                     'dist_headway', 'time_headway', 'global_heading']
        if len(self.df.columns) == 19:
            self.df.columns = col_names
        elif len(self.df.columns) == 18:
            self.df[18] = None
            self.df.columns = col_names
        for (dfind, carid) in tqdm(enumerate(self.df['id'])):
            if carid not in self.car2start:
                self.car2start[carid] = dfind
            frame = int(self.df.loc[dfind, 'frame'])
            if frame not in self.frame2cars:
                self.frame2cars[frame] = [carid]
            else:
                self.frame2cars[frame].append(carid)
        print("Finish data set initialization!")
        self.nframes = max(self.frame2cars.keys())


def carsinframe(trajdata: NGSIMTrajdata, frame: int):
    if frame not in trajdata.frame2cars:
        return []
    return trajdata.frame2cars[frame]


def carid_set(trajdata: NGSIMTrajdata):
    return set(trajdata.car2start.keys())


def nth_carid(trajdata: NGSIMTrajdata, frame: int, n: int):
    return trajdata.frame2cars[frame][n-1]


def first_carid(trajdata: NGSIMTrajdata, frame: int):
    return nth_carid(trajdata, frame, 1)


def iscarinframe(trajdata: NGSIMTrajdata, carid: int, frame: int):
    return carid in carsinframe(trajdata, frame)

    # given frame and carid, find index of car in trajdata
    # Returns 0 if it does not exist


def car_df_index(trajdata: NGSIMTrajdata, carid: int, frame: int):
    '''
    given frame and carid, find index of car in trajdata
    Returns 0 if it does not exist
    '''
    df = trajdata.df
    lo = trajdata.car2start[carid]
    framestart = df.loc[lo, 'frame']

    retval = -1

    if framestart == frame:
        retval = lo
    elif frame >= framestart:
        retval = frame - framestart + lo
        n_frames = df.loc[lo, 'n_frames_in_dataset']
        if retval > lo + n_frames:
            retval = -1

    return retval


def get_frame_range(trajdata: NGSIMTrajdata, carid: int):
    lo = trajdata.car2start[carid]
    framestart = trajdata.df.loc[lo, 'frame']

    n_frames = trajdata.df.loc[lo, 'n_frames_in_dataset']
    frameend = framestart + n_frames  # in julia there us a -1 but since python's range doesn't include end index
    return range(framestart, frameend)


# def pull_vehicle_headings(trajdata: NGSIMTrajdata, v_cutoff: float = 2.5, smoothing_width: float = 0.5):
#     df = trajdata.df
#
#     for carid in carid_set(trajdata):
#         frames = [i for i in get_frame_range(trajdata, carid)] TODO: if needed, to be updated


def load_ngsim_trajdata(filepath: str):
    return NGSIMTrajdata(filepath)




