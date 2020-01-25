module PlanningWithAttitude

using StaticArrays
using LinearAlgebra
using Parameters
using ForwardDiff
using Logging
using Formatting

include("logger.jl")
include("knotpoint.jl")
include("models.jl")
include("quaternions.jl")
include("objective.jl")
include("ilqr_solver.jl")
include("ilqr.jl")
include("attitude_costs.jl")

end
