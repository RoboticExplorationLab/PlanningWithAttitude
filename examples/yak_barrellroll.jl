
using TrajectoryOptimization
const TO = TrajectoryOptimization
using BenchmarkTools
using TrajOptPlots
using LinearAlgebra
using StaticArrays
using FileIO
using DataFrames
using MeshCat
using GeometryTypes
using CoordinateTransformations
using PlanningWithAttitude

# Start visualizer
Rot = MRP{Float64}
model = Dynamics.YakPlane(Rot)
if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
end
delete!(vis)
set_mesh!(vis, model)

function gen_barrellroll(Rot; kwargs...)
    prob = Problems.YakProblems(Rot, scenario=:barrellroll; kwargs...)
    solver = iLQRSolver(prob)
    solver.opts.verbose = false
    solver
end

solver = gen_barrellroll(UnitQuaternion{Float64,CayleyMap}, costfun=:QuatLQR)
solver = gen_barrellroll(RPY{Float64}, costfun=:Quadratic)
solver = gen_barrellroll(UnitQuaternion{Float64,IdentityMap}, use_rot=false, costfun=:Quadratic)
solve!(solver)
iterations(solver)
visualize!(vis, solver)
cost(solver)

data_br = run_all(gen_barrellroll, samples=10, evals=1)

df = DataFrame(data_br)

df[:,[:name,:costfun,:cost,:iterations,:time]]

vis = Visualizer()
open(vis)
waypoints!(vis, solver.model, solver.Z, length=15)
@save "barrell_roll.jld2" data
@load "barrell_roll.jld2"
data


label = ["Quat","RPY","iMLQR"]
p = @pgf Axis(
    {
        ybar,
        ylabel="solve time (ms)",
        # "enlarge limits" = 0.20,
        legend_style =
        {
            at = Coordinate(0.5, -0.07),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=label,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
        "every node near coord/.append style={/pgf/number format/.cd, fixed,precision=0}"
    },
    Plot(Coordinates(label, df.time)),
    # Legend(["Quadratic", "Error Quadratic", "Geodesic"])
)
pgfsave("figs/barrellroll_timing.tikz", p, include_preamble=false)
