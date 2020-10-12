import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()
using RobotDynamics
import RobotZoo.Quadrotor
using TrajectoryOptimization
# using PlanningWithAttitude
using Altro
# using TrajOptPlots
using BenchmarkTools
const TO = TrajectoryOptimization
const RD = RobotDynamics

using Random
using StaticArrays
using LinearAlgebra
# using MeshCat
using Rotations



## Test constraints
model = Quadrotor()
qf = expm(SA[0,0,1]*deg2rad(135))
x, = rand(model)
xf = RD.build_state(model, [2,3,1], qf, zeros(3), zeros(3))
con1 = QuatGeoCon(13,qf,SA[4,5,6,7])
con2 = QuatGeoCon(13,-qf,SA[4,5,6,7])
TO.evaluate(con1, x)[1] ≈ TO.evaluate(con2, x)[1]
TO.evaluate(con1, xf)[1] ≈ 0
jac = TO.gen_jacobian(con1)
TO.jacobian!(jac, con1, x)

con3 = QuatErr(13, qf, SA[4,5,6,7])
con4 = QuatErr(13,-qf, SA[4,5,6,7])
TO.evaluate(con3, x) ≈ TO.evaluate(con4, x)
TO.evaluate(con3, xf) ≈ zeros(3)
jac = TO.gen_jacobian(con3)
TO.jacobian!(jac, con3, x)

con5 = QuatVecEq(13, qf, SA[4,5,6,7])
con6 = QuatVecEq(13,-qf, SA[4,5,6,7])
TO.evaluate(con5, x) ≈ TO.evaluate(con6, x)
TO.evaluate(con5, xf) ≈ zeros(3) 
jac = TO.gen_jacobian(con5)
TO.jacobian!(jac, con5, x)

##
model = Quadrotor()
N = 51
tf = 5.0
dt = tf / (N-1)

# Objective
x0,u0 = zeros(model) 
Q = [fill(1,3), fill(0.1, 6)]
R = fill(1e-2,4)
costfun = TO.LieLQRCost(RD.LieState(model), Q, R, xf)
costfun_term = TO.LieLQRCost(RD.LieState(model), Q .* 100, R, xf)
obj = Objective(costfun, costfun_term, N)

# Constraints
cons = ConstraintList(size(model)...,N)
con0 = GoalConstraint(xf, SA[5,6,7])
add_constraint!(cons, con5, N)

prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons)
solver = ALTROSolver(prob, show_summary=true)
solve!(solver)

qf_sol = RBState(model, states(solver)[end]).q
norm(qf_sol ⊖ qf)