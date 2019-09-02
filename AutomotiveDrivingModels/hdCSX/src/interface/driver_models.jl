export
    DriverModel,

    StaticDriver,

    get_name,
    action_type,
    set_desired_speed!,
    observe!,
    reset_hidden_state!,
    prime_with_history!

abstract type DriverModel{DriveAction} end

get_name(::DriverModel) = "???"
action_type(::DriverModel{A}) where {A} = A
set_desired_speed!(model::DriverModel, v_des::Float64) = model # do nothing by default
reset_hidden_state!(model::DriverModel) = model # do nothing by default
observe!(model::DriverModel, scene::EntityFrame{S,D,I}, roadway::R, egoid::Int) where {S,D,I,R} = model  # do nothing by default
Base.rand(model::DriverModel) = error("rand not implemented for model $model")
Distributions.pdf(model::DriverModel{A}, a::A) where {A} = error("pdf not implemented for model $model")
Distributions.logpdf(model::DriverModel{A}, a::A) where {A} = error("logpdf not implemented for model $model")

function prime_with_history!(
    model::DriverModel,
    trajdata::ListRecord{Entity{S,D,I}},
    roadway::R,
    frame_start::Int,
    frame_end::Int,
    egoid::I,
    scene::EntityFrame{S,D,I} = allocate_frame(trajdata),
    ) where {S,D,I,R}

    reset_hidden_state!(model)

    for frame in frame_start : frame_end
        get!(scene, trajdata, frame)
        observe!(model, scene, roadway, egoid)
    end

    return model
end
function prime_with_history!(model::DriverModel, rec::EntityQueueRecord{S,D,I}, roadway::R, egoid::I;
    pastframe_start::Int=1-nframes(rec),
    pastframe_end::Int=0,
    ) where {S,D,I,R}

    reset_hidden_state!(model)

    for pastframe in pastframe_start : pastframe_end
        scene = rec[pastframe]
        observe!(model, scene, roadway, egoid)
    end

    model
end

####


struct StaticDriver{A,P<:ContinuousMultivariateDistribution} <: DriverModel{A}
    distribution::P
end

get_name(::StaticDriver) = "StaticDriver"
function Base.rand(model::StaticDriver{A,P}) where {A,P}
    a = rand(model.distribution)
    return convert(A, a)
end
Distributions.pdf(model::StaticDriver{A}, a::A) where {A} = pdf(model.distribution, convert(Vector{Float64}, a))
Distributions.logpdf(model::StaticDriver{A}, a::A) where {A} = logpdf(model.distribution, convert(Vector{Float64}, a))