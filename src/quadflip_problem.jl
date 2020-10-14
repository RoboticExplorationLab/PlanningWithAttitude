using RobotZoo: Quadrotor

function QuadFlipProblem(Rot=UnitQuaternion; slack::Bool=false, vecmodel::Bool=false, 
        renorm::Bool=false, costfun=LQRCost, integration=RD.RK3, termcon=:none, kwargs...)
    model = Quadrotor{Rot}()
    if renorm
        model = QuatRenorm(model)
    end
    n,m = size(model)
    m += slack
    rsize = n-9

    # discretization
    N = 101 # number of knot points
    tf = 5.0
    dt = tf/(N-1) # total time

    # Initial condition
    x0_pos = @SVector [0., -1., 1.]
    x0 = RobotDynamics.build_state(model, x0_pos, UnitQuaternion(I), zeros(3), zeros(3))

    # cost
    Q_diag = RobotDynamics.build_state(model, 
        [1e-2,1e-2,5e-2], 
        fill(1e-5,rsize), 
        fill(1e-3,3), 
        fill(1e-2,3)
    )
    R = Diagonal(@SVector fill(1e-3,m))
    q_nom = UnitQuaternion(I)
    v_nom, ω_nom = zeros(3), zeros(3)
    x_nom = RobotDynamics.build_state(model, zeros(3), q_nom, v_nom, ω_nom)

    # waypoints
    ex = @SVector [1.0, 0, 0]
    wpts = [
        ((@SVector [0, 0.5, 1.5,]),   expm(ex*deg2rad(90))),
        # ((@SVector [0, 0.2, 1.5]),    expm(ex*deg2rad(90))),
        # ((@SVector [0, 0.0, 2.0]),    expm(ex*deg2rad(135))),
        ((@SVector [0, 0.0, 2.5]),    expm(ex*deg2rad(180))),
        # ((@SVector [0, 0.0, 2.0]),    expm(ex*deg2rad(225))),
        ((@SVector [0,-0.5, 1.5]),    expm(ex*deg2rad(-90))),
        ((@SVector [0, 0.65, 1.0]),    expm(ex*deg2rad(360))),
        ((@SVector [0, 1.0, 1.0]),    expm(ex*deg2rad(360))),
    ]
    # times = [35, 41, 47, 51, 55, 61, 70, 101]
    times = [45, 51, 55, 75, 101]

    """
    Costs
    """
    # intermediate costs
    Qw_diag = RobotDynamics.build_state(model, 
        [1e3,1e1,1e3], 
        (@SVector fill(5e4,rsize)), 
        fill(1,3), fill(10,3) 
    )
    Qf_diag = RobotDynamics.fill_state(model, 10., 100, 10, 10)
    xf = RobotDynamics.build_state(model, wpts[end][1], wpts[end][2], zeros(3), zeros(3))
    cost_nom = costfun(Diagonal(Q_diag), R, xf, w=0.02)

    # waypoint costs
    costs = map(1:length(wpts)) do i
        r,q = wpts[i]
        xg = RobotDynamics.build_state(model, r, q, v_nom, [2pi/3, 0, 0])
        if times[i] == N
            Q = Diagonal(Qf_diag)
            w = 10.
        else
            Q = Diagonal(1e-3*Qw_diag)
            w = 100.
        end
        costfun(Q, R, xg, w=1.0)
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

    # Constraints
    conSet = ConstraintList(n,m,N)
    add_constraint!(conSet, GoalConstraint(xf, SA[1,2,3,8,9,10]), N:N)
    xmin = fill(-Inf,n)
    xmin[3] = 0.0
    bnd = BoundConstraint(n,m, x_min=xmin)
    add_constraint!(conSet, bnd, 1:N-1)

    if slack
        quad = QuatSlackModel(model)
        quatcon = UnitQuatConstraint(quad)
        add_constraint!(conSet, quatcon, 1:N-1)
    else
        quad = model
    end
    if vecmodel
        quad = VecModel(quad)
    end

    # Initialization
    u0 = zeros(quad)[2] 
    U_hover = [copy(u0) for k = 1:N-1] # initial hovering control trajectory

    # Problem
    prob = Problem(quad, obj, xf, tf, x0=x0, constraints=conSet, integration=integration)
    initial_controls!(prob, U_hover)

    # Infeasible start trajectory
    RobotDynamics.build_state(model, zeros(3), UnitQuaternion(I), zeros(3), zeros(3))
    X_guess = map(1:prob.N) do k
        t = (k-1)/prob.N
        x = (1-t)*x0 + t*xf
        RobotDynamics.build_state(model, position(model, x),
            expm(2pi*t*@SVector [1.,0,0]),
            RobotDynamics.linear_velocity(model, x),
            RobotDynamics.angular_velocity(model, x))
    end
    initial_states!(prob, X_guess)

    opts = SolverOptions(;
        cost_tolerance=1e-5,
        cost_tolerance_intermediate=1e-5,
        constraint_tolerance=1e-5,
        projected_newton_tolerance=1e-2,
        iterations_outer=40,
        penalty_scaling=10.,
        penalty_initial=0.1,
        show_summary=false,
        verbose_pn=false,
        verbose=0,
        projected_newton=true, 
        kwargs...
    )
    return prob, opts
end