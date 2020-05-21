
using TrajectoryOptimization
using PlanningWithAttitude
using Statistics
using Random
using StaticArrays
using LinearAlgebra
using MeshCat
using TrajOptPlots
using DataFrames
using JLD2
using PGFPlotsX
const TO = TrajectoryOptimization
include("../problems/quadrotor_problems.jl")

model = Dynamics.Quadrotor2{UnitQuaternion{Float64,CayleyMap}}()
if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

solver = gen_quad_zigzag(RPY{Float64}, use_rot=false,
    costfun=:Quadratic, constrained=false, normcon=false)
solver = gen_quad_zigzag(UnitQuaternion{Float64,IdentityMap}, use_rot=false,
    costfun=:Quadratic, constrained=false, normcon=false)
solver = gen_quad_zigzag(UnitQuaternion{Float64,CayleyMap}, use_rot=true,
    costfun=:ErrorQuad, constrained=false, normcon=false)
solver.opts.verbose = true
solve!(solver)
iterations(solver)
visualize!(vis, solver)

plot(controls(solver))
waypoints!(vis, solver.model, solver.Z, length=31)

function log_solve!(solver, data; name=rot_type(solver), kwargs...)
    b = benchmark_solve!(solver, data; kwargs...)
    push!(data[:costfun], cost_type(solver))
    push!(data[:rotation], rot_type(solver))
    push!(data[:name], name)
end

function run_all(gen_prob;kwargs...)
    # Initial data structure
    data = Dict{Symbol,Vector}(:name=>Symbol[], :costfun=>Symbol[], :iterations=>Int[], :rotation=>Symbol[],
        :time=>Float64[], :cost=>Float64[])

    # Baseline methods
    println("Baseline methods")
    log_solve!(gen_prob(UnitQuaternion{Float64, IdentityMap}, use_rot=false), data; kwargs...)
    # log_solve!(gen_prob(RodriguesParam{Float64}, use_rot=false), data; kwargs...)
    # log_solve!(gen_prob(MRP{Float64}, use_rot=false), data; kwargs...)
    log_solve!(gen_prob(RPY{Float64}, use_rot=false), data; kwargs...)

    # Quaternion Methods
    println("Quaternion methods")
    for rmap in [CayleyMap,] #[ExponentialMap, CayleyMap, MRPMap, VectorPart]
        for costfun in [:QuatLQR, :ErrorQuad] #[:Quadratic, :QuatLQR, :ErrorQuad]
            log_solve!(gen_prob(UnitQuaternion{Float64, rmap}, costfun=costfun), data; kwargs...)
        end
    end
    return data
end

function compare_quats(gen_prob; kwargs...)
    # Initial data structure
    data = Dict{Symbol,Vector}(:name=>Symbol[], :costfun=>Symbol[], :iterations=>Int[],
        :rotation=>Symbol[], :time=>Float64[], :cost=>Float64[])

    # Treat quaternion as normal vector
    println("Quat")
    log_solve!(gen_prob(UnitQuaternion{Float64, IdentityMap}, use_rot=false),
        data; name=:Quat, kwargs...)

    # Re-normalize after disretization
    println("ReNorm")
    log_solve!(gen_prob(UnitQuaternion{Float64, ReNorm}, use_rot=false),
        data; name=:ReNorm, kwargs...)

    # Use Unit Norm Constraint
    println("NormCon")
    log_solve!(gen_prob(UnitQuaternion{Float64, IdentityMap}, use_rot=false,
        normcon=true), data; name=:NormCon, kwargs...)

    # Use Unit Norm Constrained with slack
    println("QuatSlack")
    log_solve!(gen_prob(UnitQuaternion{Float64, IdentityMap}, use_rot=:slack,
        normcon=true), data; name=:QuatSlack, kwargs...)

    for rmap in [ExponentialMap, CayleyMap, MRPMap, VectorPart]
        println(rmap)
        for costfun in [:QuatLQR, :Quadratic] #[:Quadratic, :QuatLQR, :ErrorQuad]
            log_solve!(gen_prob(UnitQuaternion{Float64, rmap}, costfun=costfun,
                normcon=false), data; kwargs...)
        end
    end
    return data
end


data_zig = run_all(gen_quad_zigzag, samples=10, evals=1)
data_quats = compare_quats(gen_quad_zigzag, samples=1, evals=1)


df = DataFrame(data_zig)
df.rots = string.(short_names.(df.rotation))
df.time_per_iter = df.time ./ df.iterations
quats = in([:ExponentialMap, :CayleyMap, :MRPMap, :VectorPart])
bases = in([:IdentityMap, :MRP, :RP, :RPY])


############################################################################################
# PLOT 1: Time per iteration
############################################################################################

tpi = df[bases.(df.rotation),:]
tpi = by(df, :rots, :time_per_iter=>minimum)
coord = Coordinates(tpi.rots, tpi.time_per_iter_minimum)
cay = df[df.name .== :CayleyMap,:]
cay[:,[:name,:costfun,:time,:time_per_iter,:iterations]]

p = @pgf Axis(
    {
        ybar,
        ylabel="time per iteration (ms)",
        enlargelimits = 0.15,
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=string.(cay.costfun),
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    Plot(Coordinates(string.(cay.costfun), cay.time_per_iter)),
    # Legend(["90 degrees", "180 degrees", "270 degrees"])
)
pgfsave("figs/zig_time_per_iter_cay.tikz", p, include_preamble=false)

p = @pgf Axis(
    {
        ybar,
        ylabel="time per iteration (ms)",
        enlargelimits = 0.15,
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=tpi.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    Plot(coord),
    # Legend(["90 degrees", "180 degrees", "270 degrees"])
)
pgfsave("figs/zig_time_per_iter.tikz", p, include_preamble=false)

base = df[bases.(df.rotation),:]
quat = df[quats.(df.rotation),:]
bar_comparison(base.rots, base.time_per_iter)
qlqr  = quat[quat.costfun .== :QuatLQR,:]
equad = quat[quat.costfun .== :ErrorQuadratic,:]
quad  = quat[quat.costfun .== :Quadratic,:]
coord_base = Coordinates(base.rots, base.time_per_iter)
coord1 = Coordinates(qlqr.rots, qlqr.time_per_iter)
coord2 = Coordinates(equad.rots, equad.time_per_iter)
coord3 = Coordinates(quad.rots, quad.time_per_iter)

p = @pgf Axis(
    {
        ybar,
        ylabel="time per iteration (ms)",
        "enlarge x limits" = 0.20,
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=qlqr.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    Plot(coord3),
    Plot(coord2),
    Plot(coord1),
    Legend(["Quadratic", "Error Quadratic", "Geodesic"])
)

############################################################################################
# PLOT 2: Iterations
############################################################################################

base = df[bases.(df.rotation),:]
quat = df[quats.(df.rotation),:]
bar_comparison(base.rots, base.iterations)
qlqr  = quat[quat.costfun .== :QuatLQR,:]
equad = quat[quat.costfun .== :ErrorQuadratic,:]
quad  = quat[quat.costfun .== :Quadratic,:]
coord_base = Coordinates(base.rots, base.iterations)
coord1 = Coordinates(qlqr.rots, qlqr.iterations)
coord2 = Coordinates(equad.rots, equad.iterations)
coord3 = Coordinates(quad.rots, quad.iterations)

df2 = [equad; qlqr; base]
df2[:,[:name,:costfun,:time]]

p = @pgf Axis(
    {
        ybar,
        ylabel="iterations",
        "enlarge x limits" = 0.20,
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=tpi.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    Plot(coord3),
    Plot(coord2),
    Plot(coord1),
    Legend(["Quadratic", "Error Quadratic", "Geodesic"])
)
pgfsave("figs/zig_quat_iters.tikz", p, include_preamble=false)

p = @pgf Axis(
    {
        ybar,
        ylabel="iterations",
        "enlarge x limits" = 0.20,
        x="1.2cm",
        legend_style =
        {
            at = Coordinate(0.5, -0.15),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=tpi.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    Plot(coord_base),
    # Legend(["Geodesic", "Error Quadratic"])
)
pgfsave("figs/zig_base_iters.tikz", p, include_preamble=false)


############################################################################################
# PLOT 3: Time
############################################################################################

base = df[bases.(df.rotation),:]
quat = df[quats.(df.rotation),:]
cay = quat[quat.name .== :CayleyMap,:]
qlqr  = quat[quat.costfun .== :QuatLQR,:]
equad = quat[quat.costfun .== :ErrorQuadratic,:]
quad  = quat[quat.costfun .== :Quadratic,:]
quad_all = df[df.costfun .== :Quadratic,:]
coord_base = Coordinates(base.rots, base.time)
coord_all = Coordinates(quad_all.rots, quad_all.time)
coord1 = Coordinates(qlqr.rots, qlqr.time)
coord2 = Coordinates(equad.rots, equad.time)
coord3 = Coordinates(quad.rots, quad.time)

p = @pgf Axis(
    {
        ybar,
        ylabel="solve time (ms)",
        # "enlarge limits" = 0.20,
        "enlarge x limits" = 0.2,
        legend_style =
        {
            at = Coordinate(0.5, -0.12),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=string.(cay.costfun),
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
        "every node near coord/.append style={/pgf/number format/.cd, fixed,precision=0}"
    },
    # Plot(coord_all),
    Plot(Coordinates(string.(cay.costfun), cay.time))
    # Plot(coord3),
    # Plot(coord2),
    # Plot(coord1),
    # Legend(["Quadratic", "Error Quadratic", "Geodesic"])
)
pgfsave("figs/zig_time_cay.tikz", p, include_preamble=false)

p = @pgf Axis(
    {
        ybar,
        ylabel="solve time (ms)",
        # "enlarge limits" = 0.20,
        "enlarge x limits" = 0.2,
        legend_style =
        {
            at = Coordinate(0.5, -0.12),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=qlqr.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
        "every node near coord/.append style={/pgf/number format/.cd, fixed,precision=0}"
    },
    # Plot(coord_all),
    Plot(coord3),
    Plot(coord2),
    Plot(coord1),
    Legend(["Quadratic", "Error Quadratic", "Geodesic"])
)
pgfsave("figs/zig_time_quats.tikz", p, include_preamble=false)

best = [base; qlqr]
p = @pgf Axis(
    {
        ybar,
        ylabel="solve time (ms)",
        # "enlarge limits" = 0.20,
        "enlarge x limits" = 0.1,
        legend_style =
        {
            at = Coordinate(0.5, -0.12),
            anchor = "north",
            legend_columns = -1
        },
        symbolic_x_coords=tpi.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
        "every node near coord/.append style={/pgf/number format/.cd, fixed,precision=1}"
    },
    # Plot(coord_all),
    Plot(Coordinates(best.rots, best.time)),
    # Legend(["Quadratic", "Error Quadratic", "Geodesic"])
)
pgfsave("figs/zig_time_best.tikz", p, include_preamble=false)

print_tex(p)

############################################################################################
# PLOT 4: Quat method comparison - time
############################################################################################

df = DataFrame(data_quats)
df.rots = string.(short_names.(df.name))
quats = by(df, :rots, :time=>minimum)


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
        symbolic_x_coords=quats.rots,
        xtick = "data",
        nodes_near_coords,
        nodes_near_coords_align={vertical},
    },
    # HLine({"ultra thick", "white"}, 35),
    Plot(Coordinates(quats.rots, quats.time_minimum)),
    # Legend(["Quadratic", "Error Quadratic", "Geodesic"])
)
pgfsave("figs/zig_qcomp.tikz", p, include_preamble=false)
