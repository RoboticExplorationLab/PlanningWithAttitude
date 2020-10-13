import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using RobotDynamics
import RobotZoo.Quadrotor
using TrajectoryOptimization
using Altro
using BenchmarkTools
const TO = TrajectoryOptimization
const RD = RobotDynamics

using Random
using StaticArrays
using LinearAlgebra
using Rotations

using PlanningWithAttitude
using JLD2
using DataFrames

results = Dict(
    :problem => Symbol[],
    :errstate => Bool[],
    :time => Float64[], 
    :iters => Int[],
    :cost => Vector{Float64}[],
    :c_max => Vector{Float64}[],
)
function add_result!(name, errstate::Bool, solver)
    J0 = cost(solver)
    c0 = max_violation(solver)
    b = benchmark_solve!(solver)
    push!(results[:problem], name)
    push!(results[:errstate], errstate)
    push!(results[:time], median(b).time / 1e6)  # ms
    push!(results[:iters], iterations(solver))
    push!(results[:cost], [J0; solver.stats.cost])
    push!(results[:c_max], [c0; solver.stats.c_max])
    return b
end
results_path = joinpath(@__DIR__, "timing_results.jld2")

"""
Barrell Roll
"""
args = (
    integration = RK3,
    termcon = :quatvec,
    projected_newton = false,
    show_summary=true
)

## Original Method
prob,opts = YakProblems(vecstate=true, costfun=:Quadratic; args...) 
solver = ALTROSolver(prob, opts)
add_result!(:barrellroll, false, solver)

## Modified Method
prob,opts = YakProblems(vecstate=false, costfun=:QuatLQR; args...) 
solver = ALTROSolver(prob, opts)
add_result!(:barrellroll, true, solver)

@save results_path results 

"""
Quadrotor Flip
"""
## Original Method
prob,opts = QuadFlipProblem(vecmodel=true)
solver = ALTROSolver(prob, opts, R_inf=1e-4, infeasible=true, show_summary=true)
add_result!(:quadflip, false, solver)

## Modified Method
prob,opts = QuadFlipProblem(vecmodel=false, renorm=true, costfun=QuatLQRCost)
solver = ALTROSolver(prob, opts, R_inf=1e-4, infeasible=true, show_summary=true)
add_result!(:quadflip, true, solver)

@save results_path results

"""
Flexible Satellite
"""
## Original Method
prob,opts = SatelliteKeepOutProblem(vecstate=true, costfun=LQRCost)
solver = ALTROSolver(prob, opts)
add_result!(:satellite, false, solver)

## Modified Method
prob,opts = SatelliteKeepOutProblem(vecstate=false, costfun=QuatLQRCost)
solver = ALTROSolver(prob, opts)
add_result!(:satellite, true, solver)

@save results_path results


## Save the table
import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using LaTeXTabulars, JLD2, DataFrames, PGFPlotsX, Printf
results_path = joinpath(@__DIR__, "timing_results.jld2")
@load results_path results

df = DataFrame(results)
df.time .= round.(df.time, digits=2)
df.prob = String.(df.problem)


mod = df[df.errstate,:]
org = df[.!df.errstate,:]
tab = vcat(map(1:3) do i
    [mod.prob[i] @sprintf("%i / %i", org.iters[i], mod.iters[i]) @sprintf("%.2f / %.2f", org.time[i], mod.time[i])]
end...)
latex_tabular(joinpath(@__DIR__,"..","paper/figures/timing_results.tex"),
    Tabular("lll"),
    [
        Rule(:top),
        ["Problem", "Iterations", "time (ms)"],
        Rule(:mid),
        tab,
        Rule(:bottom)
    ]
)

## Save the Plot
c_maxes = df[df.problem .== :barrellroll,:c_max]
for c_max in c_maxes
    c_max[isinf.(c_max)] .= c_max[1]
end
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
        Coordinates(1:length(c_maxes[1]),c_maxes[1])
    ),
    PlotInc(
        {
            color="orange",
            no_marks,
            "very thick"
        },
        Coordinates(1:length(c_maxes[2]),c_maxes[2])
    ),
    Legend("naive","modified")
)
pgfsave(joinpath(@__DIR__, "..", "paper/figures/c_max_convergence.tikz"), p, include_preamble=false)