
using TrajectoryOptimization
import TrajectoryOptimization: stage_cost, cost_expansion
using TrajOptPlots
const TO = TrajectoryOptimization

# using PlanningWithAttitude
# import PlanningWithAttitude: cost_gradient, stage_cost, cost_hessian
# include("visualization.jl")
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using Plots
using ForwardDiff
using Random
using MeshCat

qref = rand(UnitQuaternion)
q = rand(UnitQuaternion)
Q = Diagonal(@SVector fill(1.,3))
qval = SVector(q)

function mycost(q::SVector{4})
    dq = CayleyMap(qref\UnitQuaternion(q))
    0.5*dq'Q*dq
end

function mycost(g::SVector{3})
    mycost(SVector(q*CayleyMap(g)))
end

mycost(qval)
g = @SVector zeros(3)
mycost(g)

grad = ForwardDiff.gradient(mycost,g)
dq = qref\q
err = CayleyMap(dq)
err'Q*TO.jacobian(CayleyMap,qref\q)*Lmult(qref)'Lmult(q)*Vmat()' ≈ grad'
Vmat()*Lmult(dq)'TO.jacobian(CayleyMap,dq)'Q*err ≈ grad


hess = ForwardDiff.hessian(mycost,g)
hess1 = Vmat()*Lmult(dq)'TO.jacobian(CayleyMap,dq)'Q*TO.jacobian(CayleyMap,dq)*Lmult(dq)*Vmat()'
hess2 = Vmat()*Lmult(dq)'TO.∇jacobian(CayleyMap,dq,Q*err)*Lmult(dq)*Vmat()'
hess1+hess2 ≈ hess

Qω = Diagonal(@SVector fill(3.,3))
R = Diagonal(@SVector fill(2.,3))
ωref = @SVector rand(3)
cost1 = SatDiffCost(model, Qω, Q, R, SVector(qref), ωref)

ω = @SVector rand(3)
x = [ωref; qval]
u = @SVector zeros(3)
z = KnotPoint(x,u,0.1)
G = TO.state_diff_jacobian(model,x)
TO.stage_cost(cost1, x, u)
ForwardDiff.gradient(x->TO.stage_cost(cost1,x,u),x)
Qxx,Quu,Qux,Qx,Qu = cost_expansion(cost1, model, z, G)
Qxx
