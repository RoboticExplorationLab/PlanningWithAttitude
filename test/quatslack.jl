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

## Test renorm
quad = Quadrotor()
x,u = rand(quad)
x2 = discrete_dynamics(RK4,quad,x,u,0.0,0.1)
q  = orientation(quad, x)
q2 = orientation(quad, x2)
norm(q)
norm(q2)

model = QuatRenorm(quad,RK4)
x3 = discrete_dynamics(RK4, model, x, u, 0.0, 0.1)
q3 = orientation(quad, x3)

## Test model
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

## Test constraint
con = UnitQuatConstraint(model)
z = KnotPoint(x2,u2,0.1)
z2 = KnotPoint(x,u2,0.1)
@test TO.evaluate(con,z)[1] ≈ 1/s^2-1 atol=1e-9
@test TO.evaluate(con,z2)[1] ≈ 0 atol=1e-12

c(x) = TO.evaluate(con, StaticKnotPoint(z, x))
∇c = ForwardDiff.jacobian(c, z.z)
∇c2 = zeros(1,18)
TO.jacobian!(∇c2, con, z)
q1 = z.z[4:7]
@test ∇c2[4:7] ≈ 2*q1
@test ∇c2 ≈ ∇c

∇c = ForwardDiff.jacobian(c, z2.z)
TO.jacobian!(∇c2, con, z2)
@test ∇c2[4:7] ≈ 2*Rotations.params(q)
@test ∇c2 ≈ ∇c

## Test VecModel
model2 = VecModel(model)
@test dynamics(model2, x2, u2) ≈ dynamics(model, x2, u2)
@test state_diff_size(model2) == 13
@test RobotDynamics.state_diff(model2, x2, x) ≈ x2 - x
@test !(model2 isa LieGroupModel)

## Test in a problem
n,m = size(model)
N,tf = 101, 5.0

Q = Diagonal(RobotDynamics.fill_state(model, 1,10,10,10.))
R = Diagonal(SA[1,1,1,1,100.]*1e-2)
x0 = zeros(model)[1] 
xf = RobotDynamics.build_state(model, 
    RBState([2,1,0.5], expm(SA[0,0,1]*pi/4), zeros(3), zeros(3))
)
Qf = (N-1)*Q
utrim = push(zeros(quad)[2],1)
obj = LQRObjective(Q, R, Qf, xf, N, uf=utrim)

cons = ConstraintList(n,m,N)
slack = UnitQuatConstraint(model)
noquat = deleteat(SVector{13}(1:13),4) 
goal = GoalConstraint(xf, noquat)
add_constraint!(cons, slack, 1:N-1)
add_constraint!(cons, goal, N)

prob = Problem(model2, obj, xf, tf, x0=x0, constraints=cons)
initial_controls!(prob, utrim) 
rollout!(prob)
Z0 = copy(prob.Z)
solver = ALTROSolver(prob)
set_options!(solver, verbose=0, show_summary=true, projected_newton=true)
solve!(solver)
norm(RBState(model, states(solver)[end]) ⊖ RBState(model, xf))
X = states(solver)
[norm(x[4:7]) for x in X]
@test size(solver.solver_al.solver_uncon.K[1]) == (5,13)

initial_trajectory!(solver, Z0)
set_options!(solver, show_summary=false)
b1 = benchmark_solve!(solver)
iterations(solver)
minimum(b1).time/iterations(solver) / 1e6  # ms/iter

## Test new method
utrim0 = pop(utrim)
R0 = Diagonal(pop(R.diag))
obj0 = LQRObjective(Q, R0, Qf, xf, N, uf=utrim0)
cons0 = ConstraintList(size(quad)...,N)
add_constraint!(cons0, goal, N)
prob0 = Problem(quad, obj0, xf, tf, x0=x0, constraints=cons0)
initial_controls!(prob0, utrim0) 
rollout!(prob0)
Z0 = copy(prob0.Z)

solver0 = ALTROSolver(prob0, show_summary=true)
solve!(solver0)
norm(RBState(quad, states(solver0)[end]) ⊖ RBState(quad, xf))
[norm(x[4:7]) for x in states(solver0)]

initial_trajectory!(solver0, Z0)
set_options!(solver0, show_summary=false)
b0 = benchmark_solve!(solver0)
iterations(solver0)
minimum(b0).time/iterations(solver0) / 1e6  # ms/iter

## Normal quad problem
R0 = Diagonal(pop(R.diag))
utrim0 = zeros(quad)[2]
obj0 = LQRObjective(Q, R0, (N-1)*Q, xf, N, uf=utrim0)
cons0 = ConstraintList(n,m-1,N)
add_constraint!(cons0, GoalConstraint(xf), N)

prob0 = Problem(quad, obj0, xf, tf, x0=x0, constraints=cons0)
initial_controls!(prob0, utrim0)
solver0 = ALTROSolver(prob0, show_summary=true)
solve!(solver0)

## Ipopt
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

prob = Problem(model, obj, xf, tf, x0=x0, constraints=copy(cons))
initial_controls!(prob, utrim) 
prob = Problem(quad, obj0, xf, tf, x0=x0, constraints=copy(cons0))
initial_controls!(prob, utrim0) 
rollout!(prob)
TO.num_constraints(prob)
TO.add_dynamics_constraints!(prob)
nlp = TO.TrajOptNLP(prob, remove_bounds=true, jac_type=:vector)

optimizer = Ipopt.Optimizer(max_iter=1000, 
    tol=1e-4, constr_viol_tol=1e-4, dual_inf_tol=1e-4, compl_inf_tol=1e-4)
TO.build_MOI!(nlp, optimizer)
MOI.optimize!(optimizer)
max_violation(nlp)

## Visualization
using TrajOptPlots
using MeshCat, Blink
if !isdefined(Main, :vis)
    vis = Visualizer()
    open(vis, Blink.Window())
end
TrajOptPlots.set_mesh!(vis, prob.model)
visualize!(vis, nlp)
visualize!(vis, solver0)


## Try another problem
prob,opts = Altro.Problems.Quadrotor(:zigzag, MRP)
prob, = Problems.YakProblems()
TO.add_dynamics_constraints!(prob)
rollout!(prob)
nlp = TO.TrajOptNLP(prob, remove_bounds=true, jac_type=:vector)
optimizer = Ipopt.Optimizer(max_iter=500, hessian_approximation="limited-memory")
TO.build_MOI!(nlp, optimizer)
MOI.optimize!(optimizer)

x = nlp.Z.Z
λ = nlp.data.λ
D,d = nlp.data.D, nlp.data.d
H,g = nlp.data.G, nlp.data.g