
function test_controller(Rot, x0::RBState; dt=1e-4, tf=10.0, dt_cntrl=dt)
    model = Dynamics.Quadrotor2{UnitQuaternion{Float64,CayleyMap}}()
    Q_ = (200.,200,50,50)
    R = Diagonal(@SVector fill(1.0,4))
    xref = zeros(model)[1]
    uref = trim_controls(model)
    if Rot <: TrajectoryOptimization.DifferentialRotation
        model = Dynamics.Quadrotor2{UnitQuaternion{Float64,Rot}}()
        Q = Diagonal(Dynamics.fill_error_state(model, Q_...))
        cntrl = MLQR(model, dt_cntrl, Q, R, xref, uref)
        xinit = Dynamics.build_state(model, x0)
        res = simulate(model, cntrl, xinit, tf, dt=dt, w=0.0)
    elseif Rot <: Rotation
        model = Dynamics.Quadrotor2{Rot}(use_rot=false)
        Q = Diagonal(Dynamics.fill_error_state(model, Q_...))
        xref = zeros(model)[1]
        cntrl = LQR(model, dt_cntrl, Q, R, xref, uref)
        xinit = Dynamics.build_state(model, x0)
        try
            res = simulate(model, cntrl, xinit, tf, dt=dt, w=0.0)
        catch
            N = tf/dt + 1
            res = [zero(xref)*NaN for i = 1:N]
        end
    elseif Rot == SE3Tracking
        Rot = UnitQuaternion{Float64,IdentityMap}
        model = Dynamics.Quadrotor2{Rot}(use_rot=false)
        xref = zeros(model)[1]

        # Build reference trajectory
        times = range(0,tf,step=dt)
        Xref = [copy(xref) for k = 1:length(times)]
        bref = [@SVector [1,0,0.] for k = 1:length(times)]
        Xdref = [@SVector zeros(13) for k = 1:length(times)]

        # cntrl = SE3Tracking(model, Xref, Xdref, bref, collect(times))
        cntrl = HFCA(model, Xref, Xdref, bref, collect(times))
        xinit = Dynamics.build_state(model, x0)
        res = simulate(model, cntrl, xinit, tf, dt=dt, w=0.0)
    end
    res = [RBState(model,x) for x in res]
    return res
end

function calc_err(X::Vector{<:RBState}, x0::RBState)
    err = map(X) do x
        dx = x ⊖ x0
        Iq = Diagonal(@SVector [1,1,1, 1,1,1, 1,1,1, 1,1,1.])
        sqrt(dx'Iq*dx)
    end
end

function generate_ICs(model,xmin,xmax,N)
    dx = xmax-xmin
    ICs = [zero(xmin) for k = 1:N]
    for i = 1:N
        x = randbetween(xmin, xmax)
        ICs[i] = x
    end
    return ICs
end

function run_MC(ICs::Vector{<:RBState}; tf=10.0, dt=1e-4, dt_cntrl=dt,
        types=[ExponentialMap, CayleyMap, MRPMap, VectorPart,
            UnitQuaternion{Float64,IdentityMap}, MRP{Float64},
            RodriguesParam{Float64}, RPY{Float64}, SE3Tracking]) where N
    # Params
    L = length(ICs)
    P = length(types)
    Nruns = L*P
    data = Dict{Symbol,Vector}(:name=>vec([rot_type(rot) for i=1:L, rot in types]),
        :max_err=>zeros(Nruns), :avg_err=>zeros(Nruns), :term_err=>zeros(Nruns),
        :IC=>zeros(Int,Nruns))

    xref = zero(RBState)

    j = 1
    for rot in types
        for i in eachindex(ICs)
            println(rot)
            X = test_controller(rot, ICs[i], dt=dt, dt_cntrl=dt_cntrl, tf=tf)
            err = calc_err(X, xref)
            data[:max_err][j] = maximum(err)
            data[:avg_err][j] = mean(err)
            data[:term_err][j] = err[end]
            data[:name][j] = rot_type(rot)
            data[:IC][j] = i
            j += 1
        end
    end
    @save "MC_data.jld2" data
    return data
end
