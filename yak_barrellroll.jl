
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
include("plotting.jl")

# Start visualizer
Rot = MRP{Float64}
model = Dynamics.YakPlane(Rot)
if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

function gen_barrellroll(Rot; kwargs...)
    prob = Problems.YakProblems(Rot, scenario=:barrellroll; kwargs...)
    solver = iLQRSolver(prob)
    solver.opts.verbose = false
    solver
end

solver = gen_barrellroll()
solver = gen_barrellroll(UnitQuaternion{Float64,VectorPart}, costfun=:ErrorQuad)
solve!(solver)
iterations(solver)
visualize!(vis, solver)
cost(solver)

data = run_all(gen_barrellroll, samples=10, evals=1)

df = DataFrame(data)
df[:,[:name,:costfun,:cost,:iterations,:time]]

vis = Visualizer()
open(vis)
waypoints!(vis, solver.model, solver.Z, length=15)
@save "barrell_roll.jld2" data
@load "barrell_roll.jld2"
data

for i = 1:15
end
