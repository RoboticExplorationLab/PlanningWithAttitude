using TrajectoryOptimization
using RobotDynamics, Rotations
using Ipopt
using LinearAlgebra, StaticArrays
using ForwardDiff
using SparseArrays
using BenchmarkTools
import RobotZoo.Quadrotor
const TO = TrajectoryOptimization

## Define the problem
model = Quadrotor{UnitQuaternion}()
n,m = size(model)
N,tf = 51,5.0
dt = tf/(N-1)

# Objective
Q = Diagonal(RobotDynamics.fill_state(model, 1,1,10,10.)*1e-3)
R = Diagonal(SA[1,1,1,1.])
Qf = (N-1)*Q*1e3
x0 = zeros(model)[1] 
xf = RobotDynamics.build_state(model, 
    RBState([2,1,0.5], expm(SA[0,0,1]*pi/4), zeros(3), zeros(3))
)
utrim = zeros(model)[2]
obj = LQRObjective(Q, R, Qf, xf, N, uf=utrim)
prob = Problem(model, obj, xf, tf, x0=x0)
initial_controls!(prob, utrim)
rollout!(prob)
initial_controls!(prob, utrim .+ 1)
J0 = cost(prob)

# Trajectory
X = states(prob)
U = controls(prob)
Z = Float64[]
for k = 1:N-1
    append!(Z, X[k])
    append!(Z, U[k])
end
append!(Z, X[N])

## Solve with Altro
using Altro
solver = ALTROSolver(prob, show_summary=true)
solve!(solver)

## Solve with Ipopt
function dynamics_constraint_structure(n,m,N)
    P = (N-1)*n
    NN = N*n + (N-1)*m
    Gstruct = spzeros(Int,P,NN) 
    ind1 = 1:n
    ind2 = 1:n+m
    blk = collect(reshape(1:(n+m)*n, n, n+m))
    for k = 1:N-1
        Gstruct[ind1,ind2] .= blk
        blk .+= n*(n+m)
        ind1 = ind1 .+ n
        ind2 = ind2 .+ (n+m)
    end
    ind1 = 1:n
    ind2 = ind1 .+ (n+m) 
    idx = (N-1)*n*(n+m)
    for k = 1:N-1
        for (i,j) in zip(ind1,ind2)
            idx += 1
            Gstruct[i,j] = idx
        end
        ind1 = ind1 .+ n
        ind2 = ind2 .+ (n+m)
    end
    return Gstruct
end

function gen_functions(prob)
    n,m,N = size(prob)
    dt = prob.Z[1].dt
    RK = TO.integration(prob)
    times = TO.get_times(prob)

    P = (N-1)*n
    NN = N*n + (N-1)*m

    # Create quadratic cost
    obj = prob.obj
    Q = obj[1].Q
    R = obj[1].R
    Qf = obj[end].Q
    Jhess = repeat([Q.diag; R.diag], N-1)*dt
    append!(Jhess, Qf.diag)
    Jhess = Diagonal(Jhess)
    Jgrad = repeat([obj[1].q; obj[2].r], N-1)*dt
    append!(Jgrad, obj[end].q)
    Jc = [o.c for o in obj]
    Jc[1:N-1] .*= dt
    Jconst = sum(Jc)
        
    # inds for state and control
    xinds = [(1:n) .+ (k-1)*(n+m) for k = 1:N]
    uinds = [(1:m) .+ n .+ (k-1)*(n+m) for k = 1:N-1]
    xinds = SVector{n}.(xinds)
    uinds = SVector{m}.(uinds)

    # constraint jacobian sparsity
    Gstruct = dynamics_constraint_structure(n, m, N)
    r,c = TO.get_rc(Gstruct)

    G = spzeros(P,NN)
    g_ = zeros(P)
    confun!(g,x) = eval_g(x,g)

    function eval_f(x)
        0.5*dot(x,Jhess,x) + dot(Jgrad,x) + Jconst
    end

    function eval_grad_f(x, grad_f)
        mul!(grad_f, Jhess, x)
        grad_f .+= Jgrad
    end

    function eval_g(x, g)
        inds = 1:n
        for k = 1:N-1
            t = times[k] 
            x1,u1 = x[xinds[k]], x[uinds[k]]
            x2 = discrete_dynamics(RK, model, x1, u1, t, dt)
            g[inds] = x2 - x[xinds[k+1]]
            inds = inds .+ n 
        end
    end

    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            rows .= r
            cols .= c
        else
            ForwardDiff.jacobian!(G, confun!, g_, x)
            for i = 1:length(r)
                values[i] = G[r[i], c[i]]
            end
        end
    end
    return eval_f, eval_grad_f, eval_g, eval_jac_g
end

eval_f, eval_grad_f, eval_g, eval_jac_g = gen_functions(prob)

eval_f(Z) ≈ J0 

grad_f = zero(Z)
eval_grad_f(Z, grad_f)
ForwardDiff.gradient(eval_f, Z) ≈ grad_f 

# test constraints
NN = length(Z)
P = (N-1)*n
g = zeros(P)
eval_g(Z, g)
err = discrete_dynamics(RK3, model, x0, utrim .+ 1, 0.0, prob.Z[1].dt) - x0
g ≈ repeat(err, N-1) 

# test constraint jacobian
nG = (N-1)*n*(n+m) + (N-1)*n
r,c,v = zeros(nG), zeros(nG), zeros(nG)
eval_jac_g(Z,:Structure,r,c,v)
eval_jac_g(Z,:Values,r,c,v)

confun!(g, x) = eval_g(x, g)
G = zeros(P,NN)
ForwardDiff.jacobian!(G, confun!, g, Z)
G ≈ sparse(r,c,v)

# bounds
x_L = fill(-1e8, NN)
x_U = fill(+1e8, NN)
g_L = fill(0.0, P)
g_U = fill(0.0, P)
x_L[1:n] .= x0
x_U[1:n] .= x0
noquat = deleteat(SVector{13}(1:13),4) 
x_L[NN-n .+ (noquat)] .= xf[noquat]
x_U[NN-n .+ (noquat)] .= xf[noquat]
iprob = createProblem(NN, x_L, x_U, P, g_L, g_U, nG, NN*NN, eval_f, eval_g, eval_grad_f, eval_jac_g)
addOption(iprob, "hessian_approximation", "limited-memory")
addOption(iprob, "max_iter", 500)
iprob.x .= Z
status = solveProblem(iprob)

## Visualize
using TrajOptPlots
using MeshCat, Blink
if !isdefined(Main, :vis)
    vis = Visualizer()
    open(vis, Blink.Window())
end
TrajOptPlots.set_mesh!(vis, model)
xinds = [(1:n) .+ (k-1)*(n+m) for k = 1:N]
xinds = SVector{n}.(xinds)
X = [iprob.x[xi] for xi in xinds]

visualize!(vis, model, tf, X)
visualize!(vis, solver)