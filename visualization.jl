using PlanningWithAttitude
using MeshCat
using GeometryTypes
using CoordinateTransformations
using FileIO

function visualize!(vis, model::AbstractModel, Z::Traj)
    X = states(Z)
    fps = Int(round(length(Z)/Z[end].t))
    anim = MeshCat.Animation(fps)
    for k in eachindex(Z)
        atframe(anim, k) do
            x = position(model, X[k])
            q = orientation(model, X[k])
            q = Quat(q...)
            settransform!(vis["robot"], compose(Translation(x), LinearMap(q)))
        end
    end
    setanimation!(vis, anim)
    return anim
end

function set_mesh!(vis, model::AbstractModel; kwargs...)
    setobject!(vis["robot"], get_mesh!(model; kwargs...))
end

get_mesh!(::Satellite; dims=[1,1,2]*0.5) = HyperRectangle(Vec((-dims/2)...), Vec(dims...))
