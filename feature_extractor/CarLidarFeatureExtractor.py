import math
from src.Roadway.roadway import Roadway
from src.Record.record import SceneRecord
from src.Record.frame import Frame
from src.Vec import VecE2, VecSE2
from src.Basic.Vehicle import Vehicle
from src.Vec.geom.line_segment import LineSegment
from src.Vec.geom.projectile import Projectile, get_intersection_time

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
            observe(self.carlidar, scene, roadway, veh_idx)
            stop = len(self.carlidar.ranges) + idx
            self.features[idx:stop] = self.carlidar.ranges
            idx += nbeams_carlidar
            if self.extract_carlidar_rangerate:
                stop = len(self.carlidar.range_rates) + idx
                self.features[idx:stop] = self.carlidar.range_rates

        return self.features

def to_oriented_bounding_box_1(retval: ConvexPolygon, center: VecSE2.VecSE2, len: float, wid: float):

    assert len > 0
    assert wid > 0
    assert center.theta is not None
    assert center.x is not None
    assert center.y is not None

    x = VecE2.polar(len/2, center.theta)
    y = VecE2.polar(wid/2, center.theta + math.pi/2)

    C = VecSE2.convert(center)
    retval.pts[0] = x - y + C
    retval.pts[1] = x + y + C
    retval.pts[2] = -x + y + C
    retval.pts[3] = -x - y + C
    retval.npts = 4

    AutomotiveDrivingModels.ensure_pts_sorted_by_min_polar_angle!(retval)

    return retval


def to_oriented_bounding_box_2(retval: ConvexPolygon, veh: Vehicle):
    return to_oriented_bounding_box_1(retval, veh.get_center, veh.definition.length_, veh.definition.width_)


def get_edge(pts: list, i: int):
    npts = len(pts)
    a = pts[i]
    if i + 1 < npts:
        b = pts[i + 1]
    else:
        b = pts[0]
    return LineSegment(a,b)

def get_poly_edge(poly: ConvexPolygon, i: int):
    return get_edge(poly.pts, i)


def get_collision_time(ray: VecSE2.VecSE2, poly: ConvexPolygon, ray_speed: float):
    min_col_time = math.inf
    for i in range(len(poly)):
        seg = get_poly_edge(poly, i)
        col_time = get_intersection_time(Projectile(ray, ray_speed), seg)
        if col_time and col_time < min_col_time:
            min_col_time = col_time
    return min_col_time



def observe(lidar: LidarSensor, scene: Frame, roadway: Roadway, vehicle_index: int):
    state_ego = scene[vehicle_index].state
    egoid = scene[vehicle_index].id
    ego_vel = VecE2.polar(state_ego.v, state_ego.posG.theta)

    in_range_ids = set()

    for veh in scene:
        if veh.id != egoid:
            a = state_ego.posG - veh.state.posG
            distance = VecE2.norm(VecE2.VecE2(a.x, a.y))
            # account for the length and width of the vehicle by considering
            # the worst case where their maximum radius is aligned
            distance = distance - math.hypot(veh.definition.length_ / 2., veh.definition.width_ / 2.)
            if distance < lidar.max_range:
                in_range_ids.add(veh.id)

                # compute range and range_rate for each angle
    for (i, angle) in enumerate(lidar.angles):
        ray_angle = state_ego.posG.theta + angle
        ray_vec = VecE2.polar(1.0, ray_angle)
        ray = VecSE2.VecSE2(state_ego.posG.x, state_ego.posG.y, ray_angle)

        range = lidar.max_range
        range_rate = 0.0
        for veh in scene:
            # only consider the set of potentially in range vehicles
            if veh.id in in_range_ids:
                lidar.poly = to_oriented_bounding_box_2(lidar.poly, veh)

                range2 = get_collision_time(ray, lidar.poly, 1.0)  # TODO: continue finish here
                if range2 and range2 < range:
                    range = range2
                    relative_speed = VecE2.polar(veh.state.v, veh.state.posG.theta) - ego_vel
                    range_rate = VecE2.proj_(relative_speed, ray_vec)
    lidar.ranges[i] = range
    lidar.range_rates[i] = range_rate


