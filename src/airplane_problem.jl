using TrajectoryOptimization: LieLQRCost
using ForwardDiff

function YakProblems(;
        integration=RD.RK4,
        N = 101,
        vecstate=false,
        scenario=:barrellroll, 
        costfun=:Quadratic, 
        termcon=:goal,
        quatnorm=:none,
        kwargs...
    )
    model = RobotZoo.YakPlane(UnitQuaternion)

    opts = SolverOptions(
        cost_tolerance_intermediate = 1e-1,
        penalty_scaling = 100.,
        penalty_initial = 0.01;
        kwargs...
    )

    s = RD.LieState(model)
    n,m = size(model)
    rsize = size(model)[1] - 9
    vinds = SA[1,2,3,8,9,10,11,12,13]

    # Discretization
    tf = 1.25
    dt = tf/(N-1)

    if scenario == :barrellroll
        ey = @SVector [0,1,0.]

        # Initial and final condition
        p0 = MRP(0.997156, 0., 0.075366) # initial orientation
        pf = MRP(0., -0.0366076, 0.) # final orientation (upside down)

        x0 = RD.build_state(model, [-3,0,1.5], p0, [5,0,0], [0,0,0])
        utrim  = @SVector  [41.6666, 106, 74.6519, 106]
        xf = RD.build_state(model, [3,0,1.5], pf, [5,0,0], [0,0,0])

        # Objective
        Qf_diag = RD.fill_state(model, 100, 500, 100, 100.)
        Q_diag = RD.fill_state(model, 0.1, 0.1, 0.1, 0.1)
        Qf = Diagonal(Qf_diag)
        Q = Diagonal(Q_diag)
        R = Diagonal(@SVector fill(1e-3,4))
        if quatnorm == :slack
            m += 1
            R = Diagonal(push(R.diag, 1e-6))
            utrim = push(utrim, 0)
        end
        if costfun == :Quadratic
            costfun = LQRCost(Q, R, xf, utrim)
            costterm = LQRCost(Qf, R, xf, utrim)
        elseif costfun == :QuatLQR
            # costfun = LieLQRCost(s, Q, R, xf, utrim; w=0.1)
            # costterm = LieLQRCost(s, Qf, R, xf, utrim; w=200.0)
            costfun = QuatLQRCost(Q, R, xf, utrim; w=0.1)
            costterm = QuatLQRCost(Qf, R, xf, utrim; w=200.0)
        elseif costfun == :LieLQR
            costfun = LieLQR(s, Q, R, xf, utrim)
            costterm = LieLQR(s, Qf, R, xf, utrim)
        elseif costfun == :ErrorQuadratic
            costfun = ErrorQuadratic(model, Q, R, xf, utrim)
            costterm = ErrorQuadratic(model, Qf, R, xf, utrim)
        end
        obj = Objective(costfun, costterm, N)

        # Constraints
        conSet = ConstraintList(n,m,N)
        vecgoal = GoalConstraint(xf, vinds) 
        if termcon == :goal
            rotgoal = GoalConstraint(xf, SA[4,5,6,7])
        elseif termcon == :quatvec
            rotgoal = QuatVecEq(n, UnitQuaternion(pf), SA[4,5,6,7])
        elseif termcon == :quaterr
            rotgoal = QuatErr(n, UnitQuaternion(pf), SA[4,5,6,7])
        else
            throw(ArgumentError("$termcon is not a known option for termcon. Options are :goal, :quatvec, :quaterr"))
        end
        add_constraint!(conSet, vecgoal, N)
        add_constraint!(conSet, rotgoal, N)

    else
        throw(ArgumentError("$scenario isn't a known scenario"))
    end

    # Initialization
    U0 = [copy(utrim) for k = 1:N-1]

    # Use a standard model (no special handling of rotation states)
    if quatnorm == :renorm 
        model = QuatRenorm(model)
    elseif quatnorm == :slack
        model = QuatSlackModel(model)
        slackcon = UnitQuatConstraint(model)
        add_constraint!(conSet, slackcon, 1:N-1)
    end
    if vecstate
        model = VecModel(model)
    end

    # Build problem
    prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet, integration=integration)
    initial_controls!(prob, U0)
    prob, opts
end
