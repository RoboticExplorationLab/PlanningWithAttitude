# using TrajectoryOptimization: LieLQRCost
using ForwardDiff
using RobotDynamics

function YakProblems(;
        integration=RD.RK4,
        N = 101,
        vecstate=false,
        scenario=:barrellroll, 
        costfun=:Quadratic, 
        termcon=:goal,
        quatnorm=:none,
        heading=0.0,  # deg
        kwargs...
    )
    model = RobotZoo.YakPlane(UnitQuaternion)

    opts = SolverOptions(
        cost_tolerance_intermediate = 1e-1,
        penalty_scaling = 10.,
        penalty_initial = 10.;
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
        dq = expm(SA[0,0,1]*deg2rad(heading))
        pf = pf * dq

        x0 = RD.build_state(model, [-3,0,1.5], p0, [5,0,0], [0,0,0])
        utrim  = @SVector  [41.6666, 106, 74.6519, 106]
        xf = RD.build_state(model, [3,0,1.5], pf, dq * [5,0,0.], [0,0,0])

        # Xref trajectory
        x̄0 = RBState(model, x0)
        x̄f = RBState(model, xf)
        Xref = map(1:N) do k
            x̄0 + (x̄f - x̄0)*((k-1)/(N-1))
        end

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
            costfuns = map(Xref) do xref
                LQRCost(Q, R, xf, utrim)
            end
            costfun = LQRCost(Q, R, xf, utrim)
            costterm = LQRCost(Qf, R, xf, utrim)
            costfuns[end] = costterm
        elseif costfun == :QuatLQR
            costfuns = map(Xref) do xref
                QuatLQRCost(Q, R, xf, utrim, w=0.1)
            end
            costterm = QuatLQRCost(Qf, R, xf, utrim; w=200.0)
            costfuns[end] = costterm
        elseif costfun == :LieLQR
            costfun = LieLQR(s, Q, R, xf, utrim)
            costterm = LieLQR(s, Qf, R, xf, utrim)
        elseif costfun == :ErrorQuadratic
            costfuns = map(Xref) do xref
                ErrorQuadratic(model, Q, R, xref, utrim)
            end
            costfun = ErrorQuadratic(model, Q, R, xf, utrim)
            costterm = ErrorQuadratic(model, Qf, R, xf, utrim)
            costfuns[end] = costterm
        end
        obj = Objective(costfuns)

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
