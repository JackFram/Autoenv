export
    run_callback


# run callback and return whether simlation should terminate
run_callback(callback::Any, rec::EntityQueueRecord{S,D,I}, roadway::R, models::Dict{I,M}, tick::Int) where {S,D,I,R,M<:DriverModel} = error("run_callback not implemented for callback $(typeof(callback))")

function _run_callbacks(callbacks::C, rec::EntityQueueRecord{S,D,I}, roadway::R, models::Dict{I,M}, tick::Int) where {S,D,I,R,M<:DriverModel,C<:Tuple{Vararg{Any}}}
    isdone = false
    for callback in callbacks
        isdone |= run_callback(callback, rec, roadway, models, tick)
    end
    return isdone
end
function simulate!(
    ::Type{A},
    rec::EntityQueueRecord{S,D,I},
    scene::EntityFrame{S,D,I},
    roadway::R,
    models::Dict{I,M},
    nticks::Int,
    callbacks::C,
    ) where {S,D,I,A,R,M<:DriverModel,C<:Tuple{Vararg{Any}}}

    empty!(rec)
    update!(rec, scene)

    # potential early out right off the bat
    if _run_callbacks(callbacks, rec, roadway, models, 0)
        return rec
    end

    actions = Array{A}(undef, length(scene))
    for tick in 1 : nticks
        get_actions!(actions, scene, roadway, models)
        tick!(scene, roadway, actions, get_timestep(rec))
        update!(rec, scene)
        if _run_callbacks(callbacks, rec, roadway, models, tick)
            break
        end
    end

    return rec
end
function simulate!(
    rec::EntityQueueRecord{S,D,I},
    scene::EntityFrame{S,D,I},
    roadway::R,
    models::Dict{I,M},
    nticks::Int,
    callbacks::C,
    ) where {S,D,I,R,M<:DriverModel,C<:Tuple{Vararg{Any}}}

    return simulate!(Any, rec, scene, roadway, models, nticks, callbacks)
end