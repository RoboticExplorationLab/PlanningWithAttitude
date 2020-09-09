

function gen_quad_line(Rot; use_rot=Rot<:UnitQuaternion, costfun=:Quadratic)
    max_con_viol = 1.0e-8
    T = Float64
    verbose = true

    opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
    cost_tolerance=1e-4,
    iterations=300)

    opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
    opts_uncon=opts_ilqr,
    iterations=40,
    cost_tolerance=1.0e-5,
    cost_tolerance_intermediate=1.0e-4,
    constraint_tolerance=max_con_viol,
    penalty_scaling=10.,
    penalty_initial=1.)

    model = Dynamics.Quadrotor2{Rot}(use_rot=use_rot)
    # model = Dynamics.LeeQuad(Rot, use_rot=use_rot)
    # model = Dynamics.BodyFrameLeeQuad(Rot, use_rot=use_rot)
    n,m = size(model)

    # discretization
    N = 101 # number of knot points
    tf = 5.0
    dt = tf/(N-1) # total time

    # Initial condition
    x0_pos = @SVector [-10., -10., 1.]
    x0 = Dynamics.build_state(model, x0_pos, I(UnitQuaternion), zeros(3), zeros(3))
    utrim = Dynamics.trim_controls(model)

    # Final state
    xf_pos = @SVector [ 10.,  10., 1.]
    xf = Dynamics.build_state(model, xf_pos, I(UnitQuaternion), zeros(3), zeros(3))

    # cost
    costfun == :QuatLQR ? sq = 0 : sq = 1
    if n == 13
        rm_quat = @SVector [1,2,3,4,5,6,8,9,10,11,12,13]
    else
        rm_quat = @SVector [1,2,3,4,5,6,7,8,9,10,11,12]
    end
    Q_diag = Dynamics.fill_state(model, 1e-3, 1e-3*sq, 1e-3, 1e-3)
    Q = Diagonal(Q_diag)
    R = Diagonal(@SVector fill(1e-3,m))
    s_term = 100 # terminal scaling

    if costfun == :QuatLQR
        cost_nom = QuatLQRCost(Q, R, xf, utrim, w=1e-2)
        cost_term = QuatLQRCost(Q*s_term, R, xf, utrim, w=10.0)
    elseif costfun == :ErrorQuad
        cost_nom = ErrorQuadratic(model, Diagonal(Q_diag[rm_quat]), R, xf)
        cost_term = ErrorQuadratic(model, Diagonal(Q_diag[rm_quat])*s_term, R, xf)
    else
        cost_nom = LQRCost(Q, R, xf, utrim)
        cost_term = LQRCost(Q*s_term, R, xf, utrim)
    end
    obj = Objective(cost_nom,cost_term,N)

    # Initialization
    # u0 = @SVector fill(0.5*9.81/4, m)
    u0 = Dynamics.trim_controls(model)
    U_hover = [copy(u0) for k = 1:N-1] # initial hovering control trajectory

    # Problem
    prob = Problem(model, obj, xf, tf, x0=x0)
    solver = iLQRSolver(prob, opts_ilqr)
    initial_controls!(solver, U_hover)
    solver.opts.verbose = true

    return solver
end

function gen_quad_zigzag(Rot; use_rot=Rot<:UnitQuaternion, costfun=:Quadratic,
        normcon=false, constrained=false)
    model = Dynamics.Quadrotor2{Rot}(use_rot=use_rot)
    n,m = size(model)

    max_con_viol = 1.0e-3
    T = Float64
    verbose = true

    opts_ilqr = iLQRSolverOptions{T}(verbose=verbose,
        cost_tolerance=1e-4,
        iterations=300)

    opts_al = AugmentedLagrangianSolverOptions{T}(verbose=verbose,
        opts_uncon=opts_ilqr,
        iterations=40,
        cost_tolerance=1.0e-5,
        cost_tolerance_intermediate=1.0e-4,
        constraint_tolerance=max_con_viol,
        penalty_scaling=10.,
        penalty_initial=1.)

    # discretization
    N = 101 # number of knot points
    tf = 5.0
    dt = tf/(N-1) # total time

    # Initial condition
    x0_pos = @SVector [0., -10., 1.]
    x0 = Dynamics.build_state(model, x0_pos, I(UnitQuaternion), zeros(3), zeros(3))

    # cost
    costfun == :QuatLQR ? sq = 0 : sq = 1
    rm_quat = @SVector [1,2,3,4,5,6,8,9,10,11,12,13]
    Q_diag = Dynamics.fill_state(model, 1e-5, 1e-5*sq, 1e-3, 1e-3)
    Q = Diagonal(Q_diag)
    R = Diagonal(@SVector fill(1e-4,m))
    q_nom = I(UnitQuaternion)
    v_nom, ω_nom = zeros(3), zeros(3)
    x_nom = Dynamics.build_state(model, zeros(3), q_nom, v_nom, ω_nom)

    if costfun == :QuatLQR
        cost_nom = QuatLQRCost(Q, R, x_nom, w=0.0)
    elseif costfun == :ErrorQuad
        cost_nom = ErrorQuadratic(model, Diagonal(Q_diag[rm_quat]), R, x_nom)
    else
        cost_nom = LQRCost(Q, R, x_nom)
    end

    # waypoints
    wpts = [(@SVector [10,0,1.]),
            (@SVector [-10,0,1.]),
            (@SVector [0,10,1.])]
    times = [33, 66, 101]
    Qw_diag = Dynamics.fill_state(model, 1e3,1*sq,1,1)
    Qf_diag = Dynamics.fill_state(model, 10., 100*sq, 10, 10)
    xf = Dynamics.build_state(model, wpts[end], I(UnitQuaternion), zeros(3), zeros(3))

    costs = map(1:length(wpts)) do i
        r = wpts[i]
        xg = Dynamics.build_state(model, r, q_nom, v_nom, ω_nom)
        if times[i] == N
            Q = Diagonal(Qf_diag)
            w = 40.0
        else
            Q = Diagonal(1e-3*Qw_diag)
            w = 0.1
        end
        if costfun == :QuatLQR
            QuatLQRCost(Q, R, xg, w=w)
        elseif costfun == :ErrorQuad
            Qd = diag(Q)
            ErrorQuadratic(model, Diagonal(Qd[rm_quat]), R, xg)
        else
            LQRCost(Q, R, xg)
        end
    end

    costs_all = map(1:N) do k
        i = findfirst(x->(x ≥ k), times)
        if k ∈ times
            costs[i]
        else
            cost_nom
        end
    end

    obj = Objective(costs_all)

    # Initialization
    u0 = @SVector fill(0.5*9.81/4, m)
    U_hover = [copy(u0) for k = 1:N-1] # initial hovering control trajectory

    # Constaints
    conSet = ConstraintSet(n,m,N)
    if normcon
        if use_rot == :slack
            add_constraint!(conSet, QuatSlackConstraint(), 1:N-1)
        else
            add_constraint!(conSet, QuatNormConstraint(), 1:N-1)
            u0 = [u0; (@SVector [1.])]
        end
    end
    bnd = BoundConstraint(n,m, u_min=0.0, u_max=12.0)
    add_constraint!(conSet, bnd, 1:N-1)

    # Problem
    prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
    solver = iLQRSolver(prob, opts_ilqr)
    initial_controls!(solver, U_hover)
    solver.opts.verbose = true

    if constrained
        solver = AugmentedLagrangianSolver(prob, opts_al)
        solver.opts.opts_uncon.verbose = false
        solver.opts.verbose = false
    end

    return solver
end
