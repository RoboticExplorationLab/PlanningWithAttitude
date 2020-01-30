
using PGFPlotsX
using Plots
using TrajOptPlots
using MeshCat

function rot_type(solver)
    Rot = Dynamics.rotation_type(solver.model)
    if Rot <: UnitQuaternion
        return Symbol(retraction_map(Rot))
    elseif Rot <: MRP
        return :MRP
    elseif Rot <: RodriguesParam
        return :RP
    elseif Rot <: RPY
        return :RPY
    end
end

function cost_type(solver)
    costfun = typeof(solver.obj[1])
    if costfun <: QuadraticCost
        return :Quadratic
    elseif costfun <: SatDiffCost
        return :SatDiff
    elseif costfun <: QuadraticQuatCost
        return :QuatLQR
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
