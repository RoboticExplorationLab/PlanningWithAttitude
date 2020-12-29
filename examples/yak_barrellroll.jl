import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using RobotDynamics
import RobotZoo.YakPlane
using TrajectoryOptimization
using Altro
using BenchmarkTools
using LaTeXTabulars
using LaTeXStrings
const TO = TrajectoryOptimization
const RD = RobotDynamics

using Random
using StaticArrays
using LinearAlgebra
using Rotations

using PlanningWithAttitude
using JLD2
using TrajOptPlots
using MeshCat
vis = Visualizer()
open(vis)

args = (
    constraint_tolerance=1e-5,
    integration = RK4,
    termcon = :quatvec,
    projected_newton = true,
    show_summary=true,
)

##
"""
Compare Cost Functions
"""

delete!(vis)
TrajOptPlots.set_mesh!(vis, YakPlane(UnitQuaternion))
prob,opts = YakProblems(vecstate=false, costfun=:QuatLQR, heading=0; args...) 
prob,opts = YakProblems(vecstate=true, costfun=:Quadratic, heading=0; args...) 
solver = ALTROSolver(prob, opts)
solve!(solver)
visualize!(vis, YakPlane(UnitQuaternion), get_trajectory(solver))

function run_example(;vecstate=false, costfun=:Quadratic, heading=0.0, args...)
    prob,opts = YakProblems(vecstate=vecstate, costfun=costfun, heading=heading; args...) 
    solver = ALTROSolver(prob, opts)
    # b = benchmark_solve!(solver)
    solve!(solver)
    push!(cost_comparison[:costfun], costfun)
    push!(cost_comparison[:errstate], !vecstate)
    # push!(cost_comparison[:time], median(b).time)
    push!(cost_comparison[:time], solver.stats.tsolve*1e6) 
    push!(cost_comparison[:iters], iterations(solver))
    push!(cost_comparison[:cost], solver.stats.cost)
    push!(cost_comparison[:c_max], solver.stats.c_max)
    push!(cost_comparison[:is_outer], solver.stats.iteration_outer)
    push!(cost_comparison[:heading], heading)
    push!(cost_comparison[:status], status(solver))
end

cost_comparison = Dict(
    :costfun => Symbol[], 
    :errstate => Bool[], 
    :time => Float64[], 
    :iters => Int[],
    :cost => Vector{Float64}[],
    :c_max => Vector{Float64}[],
    :is_outer => Vector{Int}[],
    :heading => Float64[],
    :status => Altro.TerminationStatus[]
)

# Solve with Quadratic Cost and No Error State
for h in -60:10:60
    println("Heading = $h deg")
    run_example(vecstate=true, heading=h; args...)
    run_example(costfun=:QuatLQR, heading=h; args...)
end
# run_example(; args...)
# run_example(costfun=:LieLQR; args...)
# run_example(costfun=:ErrorQuadratic; args...)

# Append initial constraint violation for plot
prob,opts = YakProblems(;args...) 
solver = ALTROSolver(prob)
c_max = max_violation(solver)
for cvals in cost_comparison[:c_max]
    cvals[cvals .== 0] .= c_max
end

# Convert to DataFrame
using DataFrames
df = DataFrame(cost_comparison)
df.time .= round.(df.time/1e6, digits=2)
df.time[df.status .!= Altro.SOLVE_SUCCEEDED] .= typemax(eltype(df.time)) 
df.iters[df.status .!= Altro.SOLVE_SUCCEEDED] .= typemax(eltype(df.iters)) 
df1 = df[.!df.errstate,[:costfun,:errstate,:time,:iters,:heading,:status]]
df2 = df[df.errstate,[:costfun,:errstate,:time,:iters,:heading,:status]]
header = ["Heading", "Iterations", "Time (ms)"]
function comp(a::T,b::T) where T
    if a > b
        s1 = a == typemax(T) ? "--" : L"%$a"
        s2 = b == typemax(T) ? "--" : L"\mathbf{%$b}"
    elseif a <= b
        s1 = a == typemax(T) ? "--" : L"\mathbf{%$a}"
        s2 = b == typemax(T) ? "--" : L"%$b"
    end
    return s1 * " / " * s2
end
tab = map(1:length(df1.time)) do i
    [
        string(df1[i,:heading]),
        comp(df1[i,:iters], df2[i,:iters]),
        comp(df1[i,:time], df2[i,:time]),
    ]
end
latex_tabular("paper/figures/test.tex",
    Tabular("lll"),
    [
        Rule(:top),
        header,
        Rule(:mid),
        tab...,
        Rule(:bottom)
    ]
)

@save "cost_comparison.jld2" cost_comparison

## Plots
using PGFPlotsX 
@load "cost_comparison.jld2" cost_comparison

y = df.c_max[(df.costfun .== :Quadratic) .& (df.heading .== 0)][1]
y2 = df.c_max[(df.costfun .== :QuatLQR) .& (df.heading .== 0)][1]
x = 1:length(y)

p = @pgf Axis(
    {
        xlabel="iterations",
        ylabel="contraint satisfaction",
        "ymode=log",
        xmajorgrids,
        ymajorgrids,
    },
    Plot(
        {
            color="cyan",
            no_marks,
            "very thick"
        },
        Coordinates(x,y)
    ),
    PlotInc(
        {
            color="orange",
            no_marks,
            "very thick"
        },
        Coordinates(1:length(y2), y2)
    ),
    Legend("original","modified")
)
pgfsave("paper/figures/c_max_convergence.tikz", p, include_preamble=false)

# # Export the table
# using DataFrames
# df = DataFrame(cost_comparison)
# df.time .= round.(df.time/1e6, digits=2)
# tab = df[:,[:costfun,:iters,:time]]
# tab.costfun = String.(tab.costfun)
# latex_tabular("paper/figures/cost_comparison.tex",
#     Tabular("lll"),
#     [
#         Rule(:top),
#         ["Cost Function", "Iterations", "time"],
#         Rule(:mid),
#         Matrix(tab),
#         Rule(:bottom)
#     ]
# )
# # savefig(p, "figs/c_max_convergence.tikz")

# ## 
# """
# Renorm Methods
# """
# # Renormalize in discrete dynamics
# prob,opts = YakProblems(costfun=:QuatLQR, quatnorm=:renorm, N=51, integration=RK2; args...) 
# solver = ALTROSolver(prob, opts)
# solve!(solver)
# norm.(orientation.(RBState.(states(solver))))

# # Add norm slack
# prob,opts = YakProblems(costfun=:QuatLQR, quatnorm=:slack, N=101, 
#     integration=RK4; args..., termcon=:quaterr) 
# solver = ALTROSolver(prob, opts, projected_newton=false)
# solver.opts.verbose = 1
# solver.opts.penalty_scaling=100
# TO.get_constraints(solver).c_max
# max_violation(solver)
# controls(solver)[end]

# solve!(solver)
# norm.(orientation.(RBState.(states(solver))))
# TO.findmax_violation(solver)

# ilqr = Altro.get_ilqr(solver)
# TO.get_constraints(solver).convals[1]

# solver = ALTROSolver(prob, opts, show_summary=true)
# size(Altro.get_ilqr(solver).K[1]) == (4,12)  # make sure it's not using the error state
# solver.opts.verbose = 2
# solver.opts.bp_reg_initial = 1e-6
# solve!(solver)

# prob,opts = YakProblems(costfun=:ErrorQuad, integration=integration)
# solver = ALTROSolver(prob, opts, show_summary=true)
# solver = Altro.get_ilqr(solver)
# solver.opts.save_S = true
# Altro.initialize!(ilqr)
# TO.state_diff_jacobian!(solver.G, solver.model, solver.Z)
# TO.dynamics_expansion!(TO.integration(solver), solver.D, solver.model, solver.Z)
# TO.error_expansion!(solver.D, solver.model, solver.G)
# TO.cost_expansion!(solver.quad_obj, solver.obj, solver.Z, true, true)
# TO.error_expansion!(solver.E, solver.quad_obj, solver.model, solver.Z, solver.G)
# ΔV = Altro.static_backwardpass!(solver)
# solver.quad_obj[end-1].Q
# diag(solver.E[end-1].Q)
# solver.S[end-1].Q

# model = prob.model
# xf = prob.xf
# X = states(prob)
# [RD.state_diff(model, x, xf, Rotations.QuatVecMap())[4:6] for x in states(prob)]

# solve!(solver)

# x1, = rand(prob.model)
# x2, = rand(prob.model)


# data_br = run_all(gen_barrellroll, samples=10, evals=1)

# df = DataFrame(data_br)

# df[:,[:name,:costfun,:cost,:iterations,:time]]

# vis = Visualizer()
# open(vis)
# waypoints!(vis, solver.model, solver.Z, length=15)
# @save "barrell_roll.jld2" data
# @load "barrell_roll.jld2"
# data


# label = ["Quat","RPY","iMLQR"]
# p = @pgf Axis(
#     {
#         ybar,
#         ylabel="solve time (ms)",
#         # "enlarge limits" = 0.20,
#         legend_style =
#         {
#             at = Coordinate(0.5, -0.07),
#             anchor = "north",
#             legend_columns = -1
#         },
#         symbolic_x_coords=label,
#         xtick = "data",
#         nodes_near_coords,
#         nodes_near_coords_align={vertical},
#         "every node near coord/.append style={/pgf/number format/.cd, fixed,precision=0}"
#     },
#     Plot(Coordinates(label, df.time)),
#     # Legend(["Quadratic", "Error Quadratic", "Geodesic"])
# )
# pgfsave("figs/barrellroll_timing.tikz", p, include_preamble=false)
