# Zhihao Zhang
# NGSIM dataset processor dataloader

import re
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from src.Vec import VecE2, VecSE2
from src.curves import CurvePt
from src.Roadway import roadway


FLOATING_POINT_REGEX = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
METERS_PER_FOOT = 0.3048


# Trajectory data loader
class NGSIMTrajdata:
    def __init__(self, file_path: str):
        assert os.path.isfile(file_path)
        self.df = pd.read_csv(file_path, sep=" ", header=None, skipinitialspace=True)
        self.car2start = {}
        self.frame2cars = {}
        col_names = ['id', 'frame', 'n_frames_in_dataset', 'epoch', 'local_x',
                     'local_y', 'global_x', 'global_y', 'length', 'width', 'class',
                     'speed', 'acc', 'lane', 'carind_front', 'carind_rear',
                     'dist_headway', 'time_headway', 'global_heading']
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


def pull_vehicle_headings(trajdata: NGSIMTrajdata, v_cutoff: float = 2.5, smoothing_width: float = 0.5):
    df = trajdata.df

    for carid in carid_set(trajdata):
        frames = [i for i in get_frame_range(trajdata, carid)]


def load_ngsim_trajdata(filepath: str):
    return NGSIMTrajdata(filepath)


# Roadway data loader
def read_boundaries(filepath_boundaries: str):
    with open(filepath_boundaries, 'r') as fp:
        lines = fp.readlines()
        fp.close()
        for i, line in enumerate(lines):
            lines[i] = line.strip()
        assert lines[0] == 'BOUNDARIES'

        n_boundaries = int(lines[1])

        assert n_boundaries >= 0

        retval = []  # Array{Vector{VecE2}}

        line_index = 1
        for i in range(n_boundaries):
            line_index += 1
            assert lines[line_index] == "BOUNDARY {}".format(i+1)
            line_index += 1
            npts = int(lines[line_index])
            line = []  # Array{VecE2}
            for j in range(npts):
                line_index += 1
                matches = re.findall(FLOATING_POINT_REGEX, lines[line_index])
                x = float(matches[0]) * METERS_PER_FOOT
                y = float(matches[1]) * METERS_PER_FOOT
                line.append(VecE2.VecE2(x, y))
            retval.append(line)
        return retval


def read_centerlines(filepath_centerlines: str):
    with open(filepath_centerlines, 'r') as fp:
        lines = fp.readlines()
        fp.close()
        for i, line in enumerate(lines):
            lines[i] = line.strip()
        assert lines[0] == 'CENTERLINES'
        n_centerlines = int(lines[1])
        assert n_centerlines >= 0
        line_index = 1
        retval = {}
        for i in range(n_centerlines):
            line_index += 1
            assert lines[line_index] == "CENTERLINE"
            line_index += 1
            name = lines[line_index]
            line_index += 1
            npts = int(lines[line_index])
            line = []
            for j in range(npts):
                line_index += 1
                matches = re.findall(FLOATING_POINT_REGEX, lines[line_index])
                x = float(matches[0]) * METERS_PER_FOOT
                y = float(matches[1]) * METERS_PER_FOOT
                line.append(VecE2.VecE2(x, y))

            centerline = []
            theta = (line[1] - line[0]).atan()
            centerline.append(CurvePt.CurvePt(VecSE2.VecSE2(line[0].x, line[0].y, theta), 0.0))
            for j in range(1, npts - 1):
                s = centerline[j - 1].s + (line[j] - line[j - 1]).hypot()
                theta = ((line[j] - line[j - 1]).atan() + (line[j + 1] - line[j]).atan())/2  # mean angle
                centerline.append(CurvePt.CurvePt(VecSE2.VecSE2(line[j], theta), s))
            s = centerline[npts - 2].s + (line[npts - 1] - line[npts - 2]).hypot()
            theta = (line[npts - 1] - line[npts - 2]).atan()
            centerline.append(CurvePt.CurvePt(VecSE2.VecSE2(line[npts - 1], theta), s))
            retval[name] = centerline
        return retval


def read_roadway(centerline_path:str, boundary_path:str):
    centerlines = read_centerlines(centerline_path)
    boundaries = read_boundaries(boundary_path)





