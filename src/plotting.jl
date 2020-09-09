export
    rot_type,
    cost_type,
    short_names,
    bar_comparison

using PGFPlotsX
using Plots
using TrajOptPlots
using MeshCat
using CoordinateTransformations

function TrajOptPlots.visualize!(vis, X::Vector{<:RBState}, tf)
    fps = Int(floor(length(X)/tf))
    anim = MeshCat.Animation(fps)
    for k in eachindex(X)
        atframe(anim, k) do
            x = X[k].r
            q = X[k].q
            settransform!(vis["robot"], compose(Translation(x), LinearMap(Quat(q))))
        end
    end
    setanimation!(vis, anim)
    return anim
end

function rot_type(solver::TrajectoryOptimization.AbstractSolver)
    rot_type(Dynamics.rotation_type(get_model(solver)))
end

function rot_type(Rot)
    if Rot <: UnitQuaternion
        return Symbol(retraction_map(Rot))
    elseif Rot <: TrajectoryOptimization.DifferentialRotation
        return Symbol(Rot)
    elseif Rot <: MRP
        return :MRP
    elseif Rot <: RodriguesParam
        return :RP
    elseif Rot <: RPY
        return :RPY
    elseif Rot == SE3Tracking
        return :SE3
    end
end

function cost_type(solver)
    obj = TO.get_objective(solver)
    if obj isa TrajectoryOptimization.ALObjective
        costfun = typeof(obj.obj[1])
    else
        costfun = typeof(obj[1])
    end
    if costfun <: QuadraticCost
        return :Quadratic
    elseif costfun <: SatDiffCost
        return :SatDiff
    elseif costfun <: QuadraticQuatCost
        return :QuatLQR
    elseif costfun <: ErrorQuadratic
        return :ErrorQuadratic
    end
end

function short_names(s::Symbol)
    if s == :ExponentialMap
        :Exp
    elseif s == :CayleyMap
        :Cay
    elseif s == :MRPMap
        :dMRP
    elseif s == :VectorPart
        :Vec
    elseif s == :IdentityMap
        :Quat
    elseif s == :ReNorm
        :Norm
    elseif s == :NormCon
        :Con
    elseif s == :QuatSlack
        :Slack
    else
        s
    end
end

function bar_comparison(labels, data; err=zero(data), name="")
    coords = Coordinates(labels, data, yerror=err)
    @pgf Axis(
        {
            ybar,
            enlargelimits = 0.15,
            x="1.1cm",
            legend_style =
            {
                at = Coordinate(0.5, -0.15),
                anchor = "north",
                legend_columns = -1
            },
            symbolic_x_coords=labels,
            xtick = "data",
            nodes_near_coords,
            nodes_near_coords_align={vertical},
            "error bars/y dir=both",
            "error bars/y explicit",
        },
        Plot(coords),
        Legend([name])
    )
end
