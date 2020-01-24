module PlanningWithAttitude

using StaticArrays
using LinearAlgebra
using Parameters
using ForwardDiff

include("knotpoint.jl")
include("models.jl")
include("quaternions.jl")
include("objective.jl")
include("ilqr_solver.jl")
include("ilqr.jl")

end
