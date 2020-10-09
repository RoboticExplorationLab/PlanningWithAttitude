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

@save "timing_results.jld2" results 

"""
Quadrotor Flip
"""
## Original Method
prob,opts = QuadFlipProblem(vecmodel=true)
solver = ALTROSolver(prob2, opts, R_inf=1e-4, infeasible=true, show_summary=true)
solve!(solver)

## Modified Method
prob,opts = QuadFlipProblem(vecmodel=false, renorm=true, costfun=QuatLQRCost)
solver = ALTROSolver(prob, opts, R_inf=1e-4, infeasible=true, show_summary=true)
solve!(solver)

@save "timing_results.jld2" results

"""
Flexible Satellite
"""