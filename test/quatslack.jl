using Test
using PlanningWithAttitude
using Rotations
using RobotDynamics
using TrajectoryOptimization
using Altro
using StaticArrays, LinearAlgebra
using ForwardDiff
import RobotZoo.Quadrotor
const TO = TrajectoryOptimization

quad = Quadrotor()
model = QuatSlackModel(quad)
@test state_dim(quad) == 13 == state_dim(model)
@test control_dim(model) == 5
x,u = rand(quad)
r,q,v,ω = RobotDynamics.parse_state(model, x)
s = rand()
x2 = RobotDynamics.build_state(model, r, q*(1/s), v, ω)
x3 = RobotDynamics.build_state(model, r, q*s, v, ω)
u2 = push(u, s) 
@test dynamics(model, x2, u2) ≈ dynamics(quad, x, u)
@test dynamics(model, x, u2) ≈ dynamics(quad, x3, u)

con = UnitQuatConstraint(model)
z = KnotPoint(x2,u2,0.1)
z2 = KnotPoint(x,u2,0.1)
@test TO.evaluate(con,z)[1] ≈ 0 atol=1e-12
@test TO.evaluate(con,z2)[1] ≈ s^2 - 1 atol=1e-12

c(x) = TO.evaluate(con, StaticKnotPoint(z, x))
∇c = ForwardDiff.jacobian(c, z.z)
∇c2 = zeros(1,18)
TO.jacobian!(∇c2, con, z)
@test ∇c[4:7] ≈ 2*Rotations.params(q)*s
@test ∇c2 ≈ ∇c

∇c = ForwardDiff.jacobian(c, z2.z)
TO.jacobian!(∇c2, con, z2)
@test ∇c[4:7] ≈ 2*Rotations.params(q)*s*s
@test ∇c2 ≈ ∇c


n,m = size(model)
N,tf = 101, 5.0

Q = Diagonal(RobotDynamics.fill_state(model, 1,10,10,10.))
R = Diagonal(SA[1,1,1,1,100.]*1e-2)
x0 = zeros(model)[1] 
xf = RobotDynamics.build_state(model, 
    RBState([2,1,0.5], expm(SA[0,0,1]*pi/4), zeros(3), zeros(3))
)
utrim = push(zeros(quad)[2],1)
obj = LQRObjective(Q, R, (N-1)*Q, xf, N, uf=utrim)

cons = ConstraintList(n,m,N)
slack = UnitQuatConstraint(model)
add_constraint!(cons, slack, 1:N-1)

prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons)
initial_controls!(prob, utrim) 
controls(prob)
rollout!(prob)
solver = ALTROSolver(prob)
set_options!(solver, verbose=2, show_summary=true)
solve!(solver)
RBState(model, states(solver)[end]) ⊖ RBState(model, xf)
controls(solver)