using TrajectoryOptimization
using PlanningWithAttitude
using StaticArrays
using LinearAlgebra
using TrajOptPlots
using MeshCat
using DataFrames
using Random
using Distributions
using Plots
using PGFPlotsX
include("../problems/quadrotor_problems.jl" )

model = Dynamics.Quadrotor2()
if !isdefined(Main,:vis)
    vis = Visualizer(); open(vis);
    set_mesh!(vis, model)
end

# Solve Quadrotor zig-zag problem
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
Rot = UnitQuaternion{Float64,CayleyMap}
model = Dynamics.Quadrotor2{Rot}(use_rot=true, mass=.5)
dt = 1e-4
Nsim = Int(solver.tf/dt) + 1
inds = Int.(range(1,Nsim,length=solver.N))
Q = Diagonal(Dynamics.fill_error_state(model, 200, 200, 50, 50))
R = Diagonal(@SVector fill(1.0, 4))
mlqr = TVLQR(model, Q, R, Xref, Uref, dt_ref)

noise = MvNormal(Diagonal(fill(5.,6)))
model_ = Dynamics.NoisyRB(model, noise)
X = simulate(model_, mlqr, Xref[1], solver.tf, w=0.0)
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

# cntrl = SE3Tracking(model, X_, Xd_, b_, t)
kx = 2.71 + 10.0
kv = 1.01 + 3.0
kR = 3.0*Diagonal(@SVector [1,1,1.])
kO = 0.5*Diagonal(@SVector [1,1,1.])
hfca = HFCA(model, X_, Xd_, b_, t, kx=kx, kv=kv, kR=kR, kO=kO)

X2 = simulate(model, hfca, Xref[1], solver.tf, w=0.0)
res_so3 = [RBState(model, x) for x in X2[inds]]
norm(res_so3 .⊖ Xref)

visualize!(vis, model_, X2[1:100:length(X)], solver.tf)
inds = Int.(range(1,length(X),length=solver.N))

visualize!(vis, model, solver.tf, X_, X[inds], X2[inds])

# Compare controller speed
function time_controller(cntrl, X)
    @benchmark map($X) do x
        get_control($cntrl, x, 0.0)
    end
end
b1 = time_controller(mlqr, X)
b2 = time_controller(hfca, X)
length(X) / (median(b1).time / 1e9)
length(X) / (median(b2).time / 1e9)
1/ratio(judge(median(b1), median(b2))).time

# Generate Plot
waypoints!(vis, model, Xref, res_lqr, res_so3, length=41)
delete!(vis["waypoints"])

# Compare all controllers
function compare_tvlqr(Xref, Uref, dt_ref, tf; dt=1e-4, w=1.0, model_params...)
    Nsim = Int(tf/dt) + 1
    inds = Int.(range(1,Nsim,length=solver.N))
    noise = MvNormal(Diagonal(fill(w,6)))

    data = Dict{Symbol,Vector}(:name=>Symbol[], :err=>Float64[],
        :err_traj=>Vector{Float64}[])

    Rots = [ExponentialMap,CayleyMap,MRPMap,VectorPart,IdentityMap,
            MRP{Float64}, RodriguesParam{Float64}, RPY{Float64}]
    Rots = [CayleyMap,RPY{Float64}]
    for Rot in Rots
        name = rot_type(Rot)
        println(name)
        if Rot <: TrajectoryOptimization.DifferentialRotation
            Rot = UnitQuaternion{Float64,Rot}
        end
        model = Dynamics.Quadrotor2{Rot}(;model_params...)
        Q = Diagonal(Dynamics.fill_error_state(model, 200, 200, 50, 50))
        R = Diagonal(@SVector fill(1.0, 4))

        cntrl = TVLQR(model, Q, R, Xref, Uref, dt_ref)
        model_sim = Dynamics.NoisyRB(model, noise)
        X = simulate(model_sim, cntrl, Xref[1], solver.tf, w=0)
        res = [RBState(model, x) for x in X[inds]]
        err = norm.(res .⊖ Xref)
        push!(data[:name], name)
        push!(data[:err], norm(err))
        push!(data[:err_traj], err)
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

    kx = 2.71 + 9.0
    kv = 1.01 + 3.0
    kR = 3.0*Diagonal(@SVector [1,1,1.])
    kO = 0.7*Diagonal(@SVector [1,1,1.])
    cntrl = HFCA(model, X_, Xd_, b_, t, kx=kx, kv=kv, kR=kR, kO=kO)

    model_sim = Dynamics.NoisyRB(model, noise)
    X2 = simulate(model_sim, cntrl, Xref[1], solver.tf, w=0)
    res = [RBState(model, x) for x in X2[inds]]
    err = norm.(res .⊖ Xref)
    push!(data[:name], :SO3)
    push!(data[:err], norm(err))
    push!(data[:err_traj], err)
    return data
end
Random.seed!(1)

ws = [0.,1,5]
data = map(ws) do w
    data = compare_tvlqr(Xref, Uref, dt_ref, solver.tf, dt=1e-4, w=w, mass=0.5)
    data[:noise] = fill(w,length(data[:err]))
    data
end




############################################################################################
#                                      PLOTS
############################################################################################
function convert_names(s::Symbol)
    if s == :IdentityMap
        return "Quat"
    elseif s == :RPY
        return "RPY"
    elseif s == :SO3
        return "SO3"
    else
        return "MLQR"
    end
end
df = vcat(DataFrame.(data)...)
df[:,[:name,:err]]
df.label = convert_names.(df.name) .* ": w = " .* string.(df.noise)

styles = Dict(ws[1]=>"solid",ws[2]=>"dotted",ws[3]=>"dashed")
colors = Dict(:CayleyMap=>"blue",:SO3=>"red")
df.style = [styles[n] for n in df.noise]
df.color = [colors[n] for n in df.name]
df.coords = map(1:size(df,1)) do i
    @pgf Plot({
        color = df.color[i],
        style = df.style[i],
        "very thick"
    }, Coordinates(t, df.err_traj[i]))
end;
df_mlqr = df[df.name .== :CayleyMap,:]
p = @pgf Axis(df_mlqr.coords..., Legend(df_mlqr.label))
p = @pgf Axis({
        ylabel="SSE",
        xlabel="time (s)",
        legend_style =
        {
            at = Coordinate(0.02, 0.99),
            "column sep=0.3cm",
            anchor = "north west",
            legend_columns = 2
        },
    },
    df.coords..., Legend(df.label)
)
pgfsave("figs/tracking_err.tikz",p,include_preamble=false)
