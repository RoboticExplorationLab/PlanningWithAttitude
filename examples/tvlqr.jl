using PlanningWithAttitude
using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using TrajOptPlots
using MeshCat
using DataFrames
using Random
include("../problems/quadrotor_problems.jl" )

model = Dynamics.Quadrotor2()
if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

# Solve Quadrotor line problem
Rot = MRP{Float64}
solver = gen_quad_zigzag(Rot, costfun=:Quadratic, use_rot=false)
solve!(solver)
visualize!(vis, solver)

# Convert trajectory to general RBState
Xref = RBState(solver.model, solver.Z)
Uref = controls(solver)
push!(Uref, Dynamics.trim_controls(model))
dt_ref = solver.Z[1].dt

# Track with different model
Rot = UnitQuaternion{Float64,ExponentialMap}
model = Dynamics.Quadrotor2{Rot}(use_rot=true, mass=1.0)
dt = 1e-4
Nsim = Int(solver.tf/dt) + 1
inds = Int.(range(1,Nsim,length=solver.N))
Q = Diagonal(Dynamics.fill_error_state(model, 200, 200, 50, 50))
R = Diagonal(@SVector fill(1.0, 4))
cntrl = TVLQR(model, Q, R, Xref, Uref, dt_ref)

X = simulate(model, cntrl, Xref[1], solver.tf, w=4.0)
res_lqr = [RBState(model, x) for x in X[inds]]
norm(res_lqr .⊖ Xref)

visualize!(vis, model, X[1:100:length(X)], solver.tf)



# Track with SE3 controller

# get reference trajectories
b_ = map(Xref) do x
    # prefer pointing torwards the goal
    x.r - Xref[end].r + 0.01*@SVector rand(3)
end
X_ = map(Xref) do x
    Dynamics.build_state(model, x)
end
Xd_ = map(1:solver.N) do k
    dynamics(model, X_[k], Uref[k])
end
t = collect(0:dt_ref:solver.tf)

cntrl = SE3Tracking(model, X_, Xd_, b_, t)

X2 = simulate(model, cntrl, Xref[1], solver.tf, w=1.0)
res_so3 = [RBState(model, x) for x in X2[inds]]
norm(res_so3 .⊖ Xref)

visualize!(vis, model, X[1:100:length(X)], solver.tf)
inds = Int.(range(1,length(X),length=solver.N))

visualize!(vis, model, solver.tf, X_, X[inds], X2[inds])


# Generate Plot
waypoints!(vis, model, Xref, res_lqr, res_so3, length=41)

# Compare all controllers
function compare_tvlqr(Xref, Uref, dt_ref, tf; dt=1e-4, w=1.0, model_params...)
    Nsim = Int(tf/dt) + 1
    inds = Int.(range(1,Nsim,length=solver.N))

    data = Dict{Symbol,Vector}(:name=>Symbol[], :err=>Float64[])

    for Rot in [ExponentialMap,CayleyMap,MRPMap,VectorPart,IdentityMap,
            MRP{Float64}, RodriguesParam{Float64}, RPY{Float64}]
        name = rot_type(Rot)
        println(name)
        if Rot <: TrajectoryOptimization.DifferentialRotation
            Rot = UnitQuaternion{Float64,Rot}
        end
        model = Dynamics.Quadrotor2{Rot}(;model_params...)
        Q = Diagonal(Dynamics.fill_error_state(model, 200, 200, 50, 50))
        R = Diagonal(@SVector fill(1.0, 4))

        cntrl = TVLQR(model, Q, R, Xref, Uref, dt_ref)
        X = simulate(model, cntrl, Xref[1], solver.tf, w=w)
        res = [RBState(model, x) for x in X[inds]]
        err = norm(res .⊖ Xref)
        push!(data[:name], name)
        push!(data[:err], err)
    end

    # get reference trajectories
    println("SO3")
    model = Dynamics.Quadrotor2(;model_params...)
    b_ = map(Xref) do x
        # prefer pointing torwards the goal
        x.r - Xref[end].r + 0.01*@SVector rand(3)
    end
    X_ = map(Xref) do x
        Dynamics.build_state(model, x)
    end
    Xd_ = map(1:solver.N) do k
        dynamics(model, X_[k], Uref[k])
    end
    t = collect(0:dt_ref:solver.tf)

    cntrl = SE3Tracking(model, X_, Xd_, b_, t)

    X2 = simulate(model, cntrl, Xref[1], solver.tf, w=w)
    res = [RBState(model, x) for x in X2[inds]]
    err = norm(res .⊖ Xref)
    push!(data[:name], :SO3)
    push!(data[:err], err)
    return data
end
Random.seed!(1)
data = compare_tvlqr(Xref, Uref, dt_ref, solver.tf, dt=1e-4, w=4.0, mass=1.0)

df = DataFrame(data)
