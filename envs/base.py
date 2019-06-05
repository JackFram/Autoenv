from envs.utils import dict_get, max_n_objects, fill_infos_cache, sample_trajdata_vehicle, load_ngsim_trajdatas
from src.Record.frame import Frame
from src.Record.record import SceneRecord, get_scene
from feature_extractor.utils import build_feature_extractor
import copy


class AutoEnv:
    def __init__(self, params: dict, trajdatas: list = None, trajinfos: list = None, roadways: list = None,
                 reclength: int = 5, delta_t: float = .1, primesteps: int = 50, H: int = 50,
                 terminate_on_collision: bool = True, terminate_on_off_road: bool = True,
                 render_params: dict = {"zoom": 5., "viz_dir": "/tmp"}):
        '''

        trajdatas::Vector{ListRecord}
        trajinfos::Vector{Dict}
        roadways::Vector{Roadway}
        roadway::Union{Nothing, Roadway} # current roadway
        scene::Scene
        rec::SceneRecord
        ext::MultiFeatureExtractor
        egoid::Int # current id of relevant ego vehicle
        ego_veh::Union{Nothing, Vehicle} # the ego vehicle
        traj_idx::Int # current index into trajdatas
        t::Int # current timestep in the trajdata
        h::Int # current maximum horizon for egoid
        H::Int # maximum horizon
        primesteps::Int # timesteps to prime the scene
        Δt::Float64

        # settings
        terminate_on_collision::Bool
        terminate_on_off_road::Bool

        # metadata
        epid::Int # episode id
        render_params::Dict # rendering options
        infos_cache::Dict # cache for infos intermediate results

        '''
        param_keys = params.keys()
        assert "trajectory_filepaths" in param_keys

        # optionally overwrite defaults
        reclength = dict_get(params, "reclength", reclength)
        primesteps = dict_get(params, "primesteps", primesteps)
        H = dict_get(params, "H", H)

        for (k, v) in dict_get(params, "render_params", render_params):
            render_params[k] = v

        terminate_on_collision = dict_get(params, "terminate_on_collision", terminate_on_collision)
        terminate_on_off_road = dict_get(params, "terminate_on_off_road", terminate_on_off_road)

        if trajdatas is None or trajinfos is None or roadways is None:
            trajdatas, trajinfos, roadways = load_ngsim_trajdatas(
                params["trajectory_filepaths"],
                minlength=primesteps + H
            )

        # build components
        scene_length = max_n_objects(trajdatas)
        scene = Frame()
        scene.init(scene_length)
        rec = SceneRecord()
        rec.init(reclength, delta_t, scene_length)
        ext = build_feature_extractor(params)
        infos_cache = fill_infos_cache(ext)

        self.trajdatas = trajdatas
        self.trajinfos = trajinfos
        self.roadways = roadways
        self.roadway = None
        self.scene = scene
        self.rec = rec
        self.ext = ext
        self.egoid = 0
        self.ego_veh = None
        self.traj_idx = 0
        self.t = 0
        self.h = 0
        self.H = H
        self.primesteps = primesteps
        self.delta_t = delta_t
        self.terminate_on_collision = terminate_on_collision
        self.terminate_on_off_road = terminate_on_off_road
        self.epid = 0
        self.render_params = render_params
        self.infos_cache = infos_cache

    def reset(self, offset: int = None, egoid: int = None, start: int = None, traj_idx: int = 1):
        if offset is None:
            offset = self.H + self.primesteps
        self.traj_idx, self.egoid, self.t, self.h = sample_trajdata_vehicle(
            self.trajinfos,
            offset,
            traj_idx,
            egoid,
            start
        )

        self.epid += 1

        self.rec.empty()
        self.scene.empty()

        # prime
        for t in range(self.t, (self.t + self.primesteps + 1)):
            self.scene = get_scene(self.scene, self.trajdatas[self.traj_idx], t)
            self.rec.update(self.scene)

        # set the ego vehicle
        self.ego_veh = self.scene[self.scene.findfirst(self.egoid)]
        # set the roadway
        self.roadway = self.roadways[self.traj_idx]
        # self.t is the next timestep to load
        self.t += self.primesteps + 1
        # enforce a maximum horizon 
        self.h = min(self.h, self.t + self.H)
        return self.get_features()

    def get_features(self):
        veh_idx = self.scene.findfirst(self.egoid)
        self.ext.pull_features(
            self.rec,
            self.roadway,
            veh_idx
        )
        return copy.deepcopy(self.ext.features)





