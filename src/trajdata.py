# Zhihao Zhang
# NGSIM dataset processor trajdata.py file

import math
import os
import numpy as np
from src import ngsim_trajdata
from src import trajectory_smoothing
from src.Vec import VecSE2
from src import const
from src.Roadway import roadway
from src.Record import record
from src.Basic import Vehicle
from tqdm import tqdm
from src.Record.record import read_trajdata


def symmetric_exponential_moving_average(arr: list, T: float, dt: float = 0.1):
    delta = T / dt
    N = len(arr)
    retval = []
    for i in range(N):
        Z = 0.0
        x = 0.0

        D = min(int(round(3 * delta)), i)

        if i + D > N - 1:
            D = N - i - 1

        for k in range(i - D, i + D + 1):
            e = math.exp(-abs(i-k)/delta)
            Z += e
            x += arr[k] * e

        retval.append(x / Z)

    return retval


class FilterTrajectoryResult:
    def __init__(self, trajdata: ngsim_trajdata.NGSIMTrajdata, carid: int):
        dfstart = trajdata.car2start[carid]
        N = trajdata.df.at[dfstart, 'n_frames_in_dataset']
        x_arr = []
        y_arr = []
        theta_arr = []
        v_arr = []
        for i in range(N):
            x_arr.append(trajdata.df.at[dfstart + i, 'global_x'])
            y_arr.append(trajdata.df.at[dfstart + i, 'global_y'])
        theta_arr.append(math.atan2(y_arr[4] - y_arr[0], x_arr[4] - x_arr[0]))
        v_arr.append(trajdata.df.at[dfstart, 'speed'])
        # hypot(ftr.y_arr[lookahead] - y₀, ftr.x_arr[lookahead] - x₀)/ν.Δt
        if v_arr[0] < 1.0:  # small speed
            # estimate with greater lookahead
            theta_arr[0] = math.atan2(y_arr[-1] - y_arr[0], x_arr[-1] - x_arr[0])
        self.carid = carid
        self.x_arr = x_arr
        self.y_arr = y_arr
        self.theta_arr = theta_arr
        self.v_arr = v_arr

    def __len__(self):
        return len(self.x_arr)


def filter_trajectory(ftr: FilterTrajectoryResult, v: trajectory_smoothing.VehicleSystem = trajectory_smoothing.VehicleSystem()):

    mu = [ftr.x_arr[0], ftr.y_arr[0], ftr.theta_arr[0], ftr.v_arr[0]]
    sigma = 1e-1
    cov_ = np.diag([sigma * 0.01, sigma * 0.01, sigma * 0.1, sigma])

    # assume control is centered
    u = [0.0, 0.0]
    z = [None, None]

    for i in range(1, len(ftr)):

        # pull observation
        z[0] = ftr.x_arr[i]
        z[1] = ftr.y_arr[i]

        # apply extended Kalman filter
        mu, cov_ = trajectory_smoothing.EKF(v, mu, cov_, u, z)

        # strong result
        ftr.x_arr[i] = mu[0]
        ftr.y_arr[i] = mu[1]
        ftr.theta_arr.append(mu[2])
        ftr.v_arr.append(mu[3])

    return ftr


def copy(trajdata: ngsim_trajdata.NGSIMTrajdata, ftr: FilterTrajectoryResult):
    dfstart = trajdata.car2start[ftr.carid]
    N = trajdata.df.at[dfstart, 'n_frames_in_dataset']

    # copy results back to trajdata
    # print("start copying: ")
    for i in range(N):
        #print(dfstart, i, N)
        #print('global_x')
        trajdata.df.at[dfstart + i, 'global_x'] = ftr.x_arr[i]
        #print('global_y')
        trajdata.df.at[dfstart + i, 'global_y'] = ftr.y_arr[i]
        trajdata.df.at[dfstart + i, 'speed'] = ftr.v_arr[i]
        #print("speed")
        if i > 0:
            a = ftr.x_arr[i]
            b = ftr.x_arr[i-1]
            c = ftr.y_arr[i]
            d = ftr.y_arr[i-1]
            trajdata.df.at[dfstart + i, 'speed'] = math.hypot(a-b, c-d) / const.NGSIM_TIMESTEP
        else:
            a = ftr.x_arr[i + 1]
            b = ftr.x_arr[i]
            c = ftr.y_arr[i + 1]
            d = ftr.y_arr[i]
            trajdata.df.at[dfstart + i, 'speed'] = math.hypot(a-b, c-d) / const.NGSIM_TIMESTEP
        #print("global_heading")
        trajdata.df.at[dfstart + i, 'global_heading'] = ftr.theta_arr[i]

    return trajdata


def filter_given_trajectory(trajdata: ngsim_trajdata.NGSIMTrajdata, carid: int):
    '''
    :param trajdata: trajectory data file, NGSIMTrajdata object
    :param carid: the id of the car that we want to filter
    :return: a smoothed trajectory
    '''
    # Filters the given vehicle's trajectory using an Extended Kalman Filter

    ftr = FilterTrajectoryResult(trajdata, carid)

    # run pre-smoothing
    ftr.x_arr = symmetric_exponential_moving_average(ftr.x_arr, const.SMOOTHING_WIDTH_POS)
    ftr.y_arr = symmetric_exponential_moving_average(ftr.y_arr, const.SMOOTHING_WIDTH_POS)

    ftr = filter_trajectory(ftr)

    trajdata = copy(trajdata, ftr)
    # print("finish copy")

    return trajdata


def load_ngsim_trajdata(filepath: str, autofilter: bool = True):
    '''
    :param filepath: the path of the raw trajectory data file
    :param autofilter: indicates do the filter or not
    :return: the filtered(or not) data class
    '''
    print("loading from file: ")
    tdraw = ngsim_trajdata.NGSIMTrajdata(filepath)

    if autofilter and os.path.splitext(filepath)[1] == ".txt":
        print("filtering:         ")
        for carid in tqdm(ngsim_trajdata.carid_set(tdraw)):
            # print(carid)
            tdraw = filter_given_trajectory(tdraw, carid)

    return tdraw


def convert(tdraw: ngsim_trajdata.NGSIMTrajdata, roadway: roadway.Roadway):
    '''
    :param tdraw: trajectory raw data
    :param roadway: roadway class
    :return: ListRecord(), a preprocessed and integrated version of tdraw and roadway
    '''
    df = tdraw.df
    vehdefs = {}
    states = []
    frames = []

    print("convert: Vehicle definition")

    for id, dfind in tdraw.car2start.items():
        vehdefs[id] = Vehicle.VehicleDef(df.at[dfind, 'class'],
                                         df.at[dfind, 'length'] * const.METERS_PER_FOOT,
                                         df.at[dfind, 'width'] * const.METERS_PER_FOOT)

    state_ind = 0
    print("convert: frames and states")
    prev_x = {}
    prev_y = {}
    for frame in tqdm(range(1, tdraw.nframes + 1)):  # change from 1 to 0

        frame_lo = state_ind + 1
        # print("frame: {}".format(frame))
        for id in ngsim_trajdata.carsinframe(tdraw, frame):
            # print("id: {}".format(id))
            if id not in prev_x:
                prev_x[id] = 0
                prev_y[id] = 0
            dfind = ngsim_trajdata.car_df_index(tdraw, id, frame)
            assert dfind != -1
            theta = math.atan2(df.at[dfind, 'global_y'] - prev_y[id], df.at[dfind, 'global_x'] - prev_x[id])
            posG = VecSE2.VecSE2(df.at[dfind, 'global_x'] * const.METERS_PER_FOOT,
                                 df.at[dfind, 'global_y'] * const.METERS_PER_FOOT,
                                 theta)
            prev_x[id] = df.at[dfind, 'global_x']
            prev_y[id] = df.at[dfind, 'global_y']
            # print(df.at[dfind, 'global_heading'])
            speed = df.at[dfind, 'speed'] * const.METERS_PER_FOOT
            state_ind += 1
            # print(state_ind)
            state = Vehicle.VehicleState()
            state.set(posG, roadway, speed)
            states.append(record.RecordState(state, id))

        frame_hi = state_ind
        frames.append(record.RecordFrame(frame_lo, frame_hi))

    return record.ListRecord(const.NGSIM_TIMESTEP, frames, states, vehdefs)


def get_corresponding_roadway(filename: str):
    '''
    :param filename: the path of the saved roadway file
    :return: the roadway class object
    '''
    retval = None
    if "i101" in filename:
        with open(os.path.join(const.DIR, "../data/ngsim_80.txt"), "r") as fp_80:
            retval = roadway.read_roadway(fp_80)
            fp_80.close()
    elif "i80" in filename:
        with open(os.path.join(const.DIR, "../data/ngsim_101.txt"), "r") as fp_101:
            retval = roadway.read_roadway(fp_101)
            fp_101.close()
    elif "holo" in filename:
        with open(os.path.join(const.DIR, "../data/ngsim_HOLO.txt"), "r") as fp_holo:
            retval = roadway.read_roadway(fp_holo)
            fp_holo.close()
    else:
        raise ValueError("no such roadway file, check your file name")
    return retval


def convert_raw_ngsim_to_trajdatas():
    '''
    convert the raw trajectory data and roadway to a integrated version and save to a file
    :return: no return
    '''
    for filepath in const.NGSIM_TRAJDATA_PATHS:
        filename = os.path.split(filepath)[1]
        print("converting " + filename)

        roadway = get_corresponding_roadway(filename)
        print("finish loading roadway.")
        print("Start loading NGSIM trajectory data.")
        tdraw = load_ngsim_trajdata(filepath)
        print("finish loading NGSIM trajectory data.")
        print("Start converting.")
        # no problems until here
        trajdata = convert(tdraw, roadway)
        print("finish converting")
        outpath = os.path.join(const.DIR, "../data/trajdata_" + filename)
        print("save to {}".format(outpath))
        with open(outpath, "w") as fp:
            trajdata.write(fp)
            fp.close()
        print("Finish saving the file")


def load_trajdata(filepath: str):
    with open(filepath, "r") as fp:
        td = read_trajdata(fp)
    return td







