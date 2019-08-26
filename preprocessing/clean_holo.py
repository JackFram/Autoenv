import pandas as pd
import numpy as np
import os
import pickle as pk
import argparse
from src.const import DIR

data_dir = os.path.join(DIR, "../preprocessing/data")
processed_dir = os.path.join(DIR, "../preprocessing/processed_data")
final_dir = os.path.join(DIR, "../data")
lane_dir = os.path.join(DIR, "../preprocessing/lane")


def clean_data(filename: str):
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath)

    # Down sampling
    gt = df.Global_Time
    bb = np.zeros(len(gt))
    for i in range(1, len(gt)):
        bb[i] = gt[i] // 100 == gt[i - 1] // 100
    sampled_gt = gt[np.where(bb == 0)[0]]
    df = df.loc[df['Global_Time'].isin(sampled_gt)]
    columnsTitles = ["Global_X", "Global_Y"]
    df[["Global_Y", "Global_X"]] = df.reindex(columns=columnsTitles)
    df = df.loc[df['Valid'] == True]
    down_sample_fn = os.path.join(data_dir, 'holo_data_downsampled.csv')
    df.to_csv(down_sample_fn, index=False)
    df = pd.read_csv(down_sample_fn)

    # Correct vehicle frames for continuity
    used_id = set(list(df['Vehicle_ID']))
    max_l = dict()
    max_w = dict()
    for i in range(len(df)):
        r = df.iloc[i]
        v_id = r.Vehicle_ID

        if v_id not in max_w.keys():
            max_w[v_id] = r.v_Width
            max_l[v_id] = r.v_length
        else:
            max_w[v_id] = max(max_w[v_id], r.v_Width)
            max_l[v_id] = max(max_l[v_id], r.v_length)

    vehicles = dict()
    show_up = set()

    id_map = dict()  # map original lost frame veh to new id
    discard_id = set()  # veh_ids that need to be replaced
    id_cnt = 1
    r_id_map = dict()

    last_LY = dict()
    frame_cnt = dict()
    for i in range(len(df)):
        r = df.iloc[i]
        v_id = r.Vehicle_ID

        # set width and length
        df.at[i, 'v_Width'] = max_w[v_id]
        df.at[i, 'v_length'] = max_l[v_id]

        if v_id in id_map:
            v_id = id_map[v_id]
        elif v_id in discard_id:  # find a suitable id to replace dicard id
            while id_cnt in used_id or id_cnt in discard_id:
                id_cnt += 1
            id_map[v_id] = id_cnt
            r_id_map[id_cnt] = v_id
            v_id = id_map[v_id]
        # if v_id != r.Vehicle_ID:
        #     print(i, ' ', v_id, ' ', r.Vehicle_ID)
        df.at[i, 'Vehicle_ID'] = v_id

        if v_id not in vehicles.keys():
            last_LY[v_id] = 0
            frame_cnt[v_id] = 1
            df.at[i, 'v_Acc'] = 0
            df.at[i, 'v_Vel'] = 0
            df.at[i, 'Frame_ID'] = (df.at[i, 'Global_Time'] - min(df['Global_Time'])) // 100 + 1
        else:
            if df.at[i, 'Section_ID'] != df.at[vehicles[v_id], 'Section_ID']:
                last_LY[v_id] = df.at[vehicles[v_id], 'Local_Y']
            df.at[i, 'Local_Y'] += last_LY[v_id]

            df.at[i, 'v_Vel'] = np.sqrt((df.at[i, 'Global_Y'] - df.at[vehicles[v_id], 'Global_Y']) ** 2 +
                                        (df.at[i, 'Global_X'] - df.at[vehicles[v_id], 'Global_X']) ** 2)
            df.at[i, 'v_Acc'] = df.at[i, 'v_Vel'] - df.at[vehicles[v_id], 'v_Vel']

            df.at[i, 'Frame_ID'] = df.at[vehicles[v_id], 'Frame_ID'] + 1

            frame_cnt[v_id] += 1

        df.at[i, 'Lane_ID'] = int(df.at[i, 'Local_X'] // 3.75)

        show_up.add(v_id)
        if v_id in vehicles and (r.Global_Time // 100) != (df.at[vehicles[v_id], 'Global_Time'] // 100) + 1:
            vehicles.pop(v_id)
            discard_id.add(v_id)
            # id_map.pop(r_id_map[v_id])
            # print(v_id)
        vehicles[v_id] = i
    for i in range(len(df)):
        df.at[i, 'Total_Frame'] = frame_cnt[df.at[i, 'Vehicle_ID']]
    df[['Local_X', 'Local_Y', 'Global_X', 'Global_Y', 'v_Vel', 'v_Acc', 'v_length', 'v_Width', 'Space_Headway']] *= 3.28
    # df['Frame_ID'] = (df['Global_Time'] - min(df['Global_Time'])) // 100 + 1
    df['Frame_ID'] = (df['Frame_ID'] - min(df['Frame_ID'])) + 1
    df['Vehicle_ID'] += 1
    df[df['Preceding'] != 0]['Preceding'] += 1
    df[df['Following'] != 0]['Following'] += 1
    le0_ids = np.unique(df[df['Local_X'] < 0]['Vehicle_ID'])
    df = df[[k not in le0_ids for k in df['Vehicle_ID']]]
    saved_path = os.path.join(processed_dir, 'holo_{}_perfect_cleaned.csv'.format(filename[5:19]))
    print("save to {}".format(saved_path))
    df.to_csv(saved_path, index=False)
    return df.shape[0]


def csv2txt(filename: str):
    filepath = os.path.join(processed_dir, filename)
    df = pd.read_csv(filepath)
    df = df.sort_values(by=['Vehicle_ID', 'Frame_ID'])
    df = df[df["Total_Frame"] >= 5]
    dd = df[['Vehicle_ID','Frame_ID','Total_Frame','Global_Time', 'Local_X', 'Local_Y', 'Global_X', 'Global_Y', 'v_length',
       'v_Width', 'v_Class', 'v_Vel', 'v_Acc', 'Lane_ID', 'Preceding', 'Following', 'Space_Headway', 'Time_Headway']]
    save_file = os.path.join(final_dir, r'holo_trajectories.txt')
    print("save to {}".format(save_file))
    np.savetxt(save_file, dd.values, fmt='%4d %8d %8d %15d %8.3f %8.3f %14.3f %12.3f %6.3f %6.3f %3d %8.3f %8.3f  %d  %6d %6d %8.3f %8.3f')


def create_lane(filename: str):
    start = 0
    lane_df = {}
    for k in range(3):
        filepath = os.path.join(data_dir, filename)
        file_name = filepath + str(k)
        df = pd.read_csv(file_name + '_corrected_smoothed.csv')
        lane_df[k] = df['Lane_Boundary_Left_Global']

    lane_df[3] = df['Lane_Boundary_Right_Global']

    lane_cnt = 4

    for k in range(lane_cnt):
        #     plt.figure()

        lane = np.zeros((len(df), 2))
        for i in range(len(df)):
            #         print(df.at[i,'Lane_Boundary_Left_Global'][2:-2].split(']\n ['))
            a = np.array(
                list(map(lambda x: np.array(x.split()).astype(np.float) * 3.28, lane_df[k][i][2:-2].split(']\n ['))))
            lane[i, :] = a[len(a) // 2]

        lane.T[[0, 1]] = lane.T[[1, 0]]

        indexes = np.unique(lane, return_index=True, axis=0)[1]
        lane = lane[sorted(indexes)]
        print(len(lane))
        #     gap = np.array([[0,0],[0,10],[0,20]])
        #     lane = np.vstack((gap, lane))
        #     idx = np.where((461000 < lane[:,0]) & (lane[:,0] < 466000))[0]
        # plt.plot(lane[:, 0], lane[:, 1])
        # plt.savefig(file_name + '.png')
        #     plt.scatter(lane[:,0], lane[:,1])
        #     plt.savefig(file_name + '_original.png')
        f = open(os.path.join(lane_dir, ('lane' + str(k) + '.pk')), 'wb')
        pk.dump(lane, f)
    # plt.show()
    lanes = dict()
    for i in range(lane_cnt):
        f = open(os.path.join(lane_dir, ('lane' + str(i) + '.pk')), 'rb')
        lanes[i] = pk.load(f)

    centers = {}
    for l in range(lane_cnt - 1):
        j = 0
        centers[l] = np.zeros(lanes[l].shape)
        for i in range(len(lanes[l])):
            dis1 = dis2 = dis3 = 1e9
            if j > 0:
                dis1 = np.linalg.norm(lanes[l][i, :] - lanes[l + 1][j - 1, :])
            dis2 = np.linalg.norm(lanes[l][i, :] - lanes[l + 1][j, :])
            if j + 1 < len(lanes[l + 1]):
                dis3 = np.linalg.norm(lanes[l][i, :] - lanes[l + 1][j + 1, :])

            k = j
            if dis3 <= dis2 and dis3 <= dis1 and j < len(lanes[l + 1]):
                k = j + 1
                j += 1
            elif dis1 < dis2 and dis1 < dis3:
                k = j - 1

            centers[l][i, :] = (lanes[l][i, :] + lanes[l + 1][k, :]) / 2
    # for i in range(lane_cnt):
    #     plt.plot(lanes[i][:, 0], lanes[i][:, 1])
    # for i in range(lane_cnt - 1):
    #     plt.plot(centers[i][:, 0], centers[i][:, 1], linestyle=':')
    # plt.xlim(5300 + 1.453e7, 5600 + 1.453e7)
    # plt.ylim(1520000, 1522000)
    # plt.show()
    boundary_fn = os.path.join(final_dir, 'boundariesHOLO.txt')
    f = open(boundary_fn, 'wb')
    f.write(b'BOUNDARIES\n')
    f.write((str((lane_cnt - 1) * 2) + '\n').encode())
    for i in range(lane_cnt - 1):
        f.write(('BOUNDARY ' + str(2 * i + 1) + '\n').encode())
        f.write((str(len(lanes[i])) + '\n').encode())
        np.savetxt(f, lanes[i], fmt=(' %.5f %.5f'))
        f.write(('BOUNDARY ' + str(2 * i + 2) + '\n').encode())
        f.write((str(len(lanes[i + 1])) + '\n').encode())
        np.savetxt(f, lanes[i + 1], fmt=(' %.5f %.5f'))
    f.close()
    print("boundariesHOLO.txt has been saved to {}".format(boundary_fn))
    centerline_fn = os.path.join(final_dir, 'centerlinesHOLO.txt')
    f = open(centerline_fn, 'wb')
    f.write(b'CENTERLINES\n')
    f.write((str(lane_cnt - 1) + '\n').encode())
    for i in range(lane_cnt - 1):
        f.write(('CENTERLINE\n').encode())
        f.write(('centerline' + str(i + 1) + '\n').encode())
        f.write((str(len(lanes[i])) + '\n').encode())
        np.savetxt(f, centers[i], fmt=(' %.5f %.5f'))
    f.close()
    print("centerlinesHOLO.txt has been saved to {}".format(centerline_fn))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='validation settings')
    parser.add_argument('--traj_path', type=str, default=None)
    parser.add_argument('--lane_path', type=str, default=None)

    clean_args = parser.parse_args()
    if clean_args.traj_path is None:
        raise ValueError("You need to input a raw trajectory data path")

    clean_data(clean_args.traj_path)
    processed_data_path = 'holo_{}_perfect_cleaned.csv'.format(clean_args.traj_path[5:19])
    csv2txt(processed_data_path)

    if clean_args.lane_path is not None:
        create_lane(clean_args.lane_path)

    print("Finish data preprocessing")


