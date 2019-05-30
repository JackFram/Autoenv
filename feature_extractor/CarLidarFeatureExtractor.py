import math
from src.Roadway.roadway import Roadway
from src.Record.record import SceneRecord

class ConvexPolygon:
    def __init__(self, npts):
        self.pts = [None for i in range(npts)]
        self.npts = 0

    def __len__(self):
        return self.npts


class LidarSensor:
    def __init__(self, nbeams: int, max_range: float = 100.0, angle_offset: float = 0.0, angle_spread: float = 2*math.pi):
        if nbeams > 1:
            start = angle_offset - angle_spread/2
            stop = angle_offset + angle_spread/2
            length = nbeams+1
            interval = (stop - start)/(length - 1)
            angles = [start+interval*i for i in range(1, length)]
        else:
            angles = []
            nbeams = 0
        ranges = [None for i in range(nbeams)]
        range_rates = [None for i in range(nbeams)]
        self.angles = angles
        self.ranges = ranges
        self.range_rates = range_rates
        self.max_range = max_range
        self.poly = ConvexPolygon(4)

    @property
    def nbeams(self):
        return len(self.angles)


class CarLidarFeatureExtractor:
    def __init__(self, carlidar_nbeams: int = 20, extract_carlidar_rangerate: bool = True, carlidar_max_range: float = 50.0):
        self.carlidar = LidarSensor(carlidar_nbeams, max_range=carlidar_max_range, angle_offset=0.)
        self.num_features = self.carlidar.nbeams * (1 + extract_carlidar_rangerate)
        self.features = [0 for i in range(self.num_features)]
        self.extract_carlidar_rangerate = extract_carlidar_rangerate

    def __len__(self):
        return self.num_features

    def pull_features(self, rec: SceneRecord, roadway: Roadway, veh_idx: int,
                      models: {}, pastframe: int = 0):
        scene = rec[pastframe]
        nbeams_carlidar = self.carlidar.nbeams
        idx = 0
        if nbeams_carlidar > 0:
            observe!(self.carlidar, scene, roadway, veh_idx)
            stop = len(self.carlidar.ranges) + idx
            self.features[idx:stop] = self.carlidar.ranges
            idx += nbeams_carlidar
            if self.extract_carlidar_rangerate:
                stop = len(self.carlidar.range_rates) + idx
                self.features[idx:stop] = self.carlidar.range_rates

        return self.features



