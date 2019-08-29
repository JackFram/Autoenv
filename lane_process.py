import pandas as pd
import numpy as np
import os
import pickle as pk


def create_lane_pk(data_file_fn="./preprocessing/data/"):
    lane_cnt = 4
    lane = {}
    for i in range(lane_cnt):
        lane[i] = np.zeros((0, 2))
    for filename in os.listdir(data_file_fn):
        if os.path.isdir(os.path.join(data_file_fn, filename, 'processed')):
            lane_df = {}
            lane_name = filename + "_lane"
            print(os.path.join(data_file_fn, filename, 'processed', lane_name))
            data_dir = os.path.join(data_file_fn, filename, 'processed')
            filename = lane_name
            for k in range(3):
                filepath = os.path.join(data_dir, filename)
                file_name = filepath + str(k)
                df = pd.read_csv(file_name + '_corrected_smoothed.csv')
                lane_df[k] = df['Lane_Boundary_Left_Global']

            lane_df[3] = df['Lane_Boundary_Right_Global']

            for k in range(lane_cnt):

                lane_ = np.zeros((len(df), 2))
                for i in range(len(df)):
                    #         print(df.at[i,'Lane_Boundary_Left_Global'][2:-2].split(']\n ['))
                    a = np.array(
                        list(map(lambda x: np.array(x.split()).astype(np.float) * 3.28,
                                 lane_df[k][i][2:-2].split(']\n ['))))
                    lane_[i, :] = a[len(a) // 2]
                    if lane_[i, 0] < 1500000:
                        print(lane_[i, :])

                lane_.T[[0, 1]] = lane_.T[[1, 0]]
                lane[k] = np.concatenate([lane[k], lane_], axis=0)
    lane_dir = "./preprocessing/lane"
    length = 10000
    for i in range(lane_cnt):
        indexes = np.unique(lane[i], return_index=True, axis=0)[1]
        print(lane[i][indexes][:10])
        lane[i] = lane[i][indexes]
        indexes = [int(m * len(lane[i]) / length) for m in range(length)]
        lane[i] = lane[i][indexes]
        max_value = -1
        for k in range(1, len(lane[i])):
            value = abs(lane[i][k, 0] - lane[i][k - 1, 0]) + abs(lane[i][k, 1] - lane[i][k - 1, 1])
            if value >= 1:
                if value > max_value:
                    max_value = value
                    print(lane[i][k, 0], lane[i][k - 1, 0])
                    print(lane[i][k, 1], lane[i][k - 1, 1])

        print("max value is {}".format(max_value))

        f = open(os.path.join(lane_dir, ('lane' + str(i) + '.pk')), 'wb')
        pk.dump(lane[i], f)


def generate_boundary(lane_dir="./preprocessing/lane"):
    lanes = dict()
    lane_cnt = 4
    final_dir = "./data"
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


def generate_roadway():
    import julia
    j = julia.Julia()
    j.using("NGSIM")
    base_dir = os.path.expanduser('~/Autoenv/data/')
    j.write_roadways_to_dxf(base_dir)
    j.write_roadways_from_dxf(base_dir)


if __name__ == "__main__":
    create_lane_pk()
    generate_boundary()
    generate_roadway()

