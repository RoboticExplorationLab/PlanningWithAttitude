export
    TVLQR,
    LQR,
    MLQR,
    simulate,
    SE3Tracking,
    test_ICs

import TrajectoryOptimization.Dynamics: trim_controls, build_state
abstract type AbstractController end
abstract type LQRController <: AbstractController end
abstract type TimeVaryingController <: AbstractController end
abstract type TrackingController <: TimeVaryingController end

import TrajectoryOptimization: state_diff, state_diff_jacobian

function get_k(cntrl::TimeVaryingController, t)
    times = get_times(cntrl)
    searchsortedlast(times,t)
end


""" TVLQR
"""
struct TVLQR{L,TK} <: TimeVaryingController
    model::L
    Z::Traj
    K::Vector{TK}
    t::Vector{Float64}
end

""" Assumes the states of the trajectory Z match the states of the model
"""
function TVLQR(model::AbstractModel, Q, R, Z::Traj)
    A,B = linearize(model, Z)
    K = tvlqr(A,B,Q,R)
    t = [z.t for z in Z]
    TVLQR(model,Z,K,t)
end

function TVLQR(model::RigidBody, Q, R, X::Vector{<:RBState}, U::Vector{<:AbstractVector}, dt)
    # Convert the trajectory to the orientation representation of the model
    Z = Traj(model, X, U, dt)

    # Call as normal
    TVLQR(model, Q, R, Z::Traj)
end

@inline get_times(cntrl::TVLQR) = cntrl.t

function get_control(cntrl::TVLQR, x, t)
    k = get_k(cntrl, t)
    zref = cntrl.Z[k]
    dx = state_diff(cntrl.model, x, state(zref))
    return cntrl.K[k]*dx + control(zref)
end

function tvlqr(A, B, Q, R)
    n,m = size(B[1])
    K = [@SMatrix zeros(m,n) for k = 1:length(A)]
    tvlqr!(K, A, B, Q, R)
end

function tvlqr!(K, A, B, Q, R)
    N = length(A)

    # Solve infinite-horizon at goal state
    Qf = dare(A[N], B[N], Q, R)

    P_ = similar_type(A[N])
    P = copy(Qf)
    for k = N-1:-1:1
        Pk = Q + A[k]'P*A[k] - A[k]'P*B[k]*((R+B[k]'P*B[k])\(B[k]'P*A[k]))
        K[k] = -(R+B[k]'P*B[k])\(B[k]'P*A[k])
        P = copy(Pk)
    end
    return K
end

""" Discrete LQR
"""
struct LQR{T,N,M,TK} <: LQRController
    K::TK
    xref::SVector{N,T}
    uref::SVector{M,T}
end

function LQR(model::AbstractModel, dt::Real, Q, R,
        xeq=zeros(model)[1], ueq=trim_controls(model))
    # Linearize the model
    A,B = linearize(model, xeq, ueq, dt)

    # Calculate the optimal control gain
    K = calc_LQR_gain(A,B,Q,R)
    LQR(K, xeq, ueq)
end

function get_control(cntrl::LQR, x, t)
    dx = x - cntrl.xref
    return cntrl.K*dx + cntrl.uref
end

""" Multiplicative LQR
"""
struct MLQR{T,N,M,TK} <: LQRController
    model::AbstractModel
    K::TK
    xref::SVector{N,T}
    uref::SVector{M,T}
end


function MLQR(model::AbstractModel, dt::Real, Q::AbstractMatrix, R::AbstractMatrix,
        xeq=zeros(model)[1], ueq=trim_controls(model))
    # Linearize the model, accounting for attitude state
    A,B = linearize(model, xeq, ueq, dt)
    G1 = state_diff_jacobian(model, xeq)
    G2 = state_diff_jacobian(model, discrete_dynamics(RK3, model, xeq, ueq, 0.0, dt))
    A = G2'A*G1
    B = G2'B

    # Calculate the optimal control gain
    @assert size(Q) == size(A)
    K = calc_LQR_gain(A,B,Q,R)
    MLQR(model,K,xeq,ueq)
end

function get_control(cntrl::MLQR, x, t)
    dx = state_diff(cntrl.model, x, cntrl.xref)
    return cntrl.K*dx + cntrl.uref
end

""" SE(3) Quadrotor controller
from [Lee 2010]
"""
struct SE3Tracking{T,N} <: TrackingController
    model::AbstractModel
    kx::T
    kv::T
    kR::T
    kΩ::T
    Xref::Vector{SVector{N,T}}
    Xdref::Vector{SVector{N,T}}
    bref::Vector{SVector{3,T}}
    t::Vector{T}
end

function SE3Tracking(model::AbstractModel, Xref, Xdref, bref, t;
        # kx=59.08, kv=24.3, kR=8.81, kO=1.54)
        kx=2.71, kv=1.01, kR=2.26, kO=0.1)
    SE3Tracking(model, kx, kv, kR, kO, Xref, Xdref, bref, t)
end

@inline get_times(cntrl::SE3Tracking) = cntrl.t

function get_control(c::SE3Tracking, x, t)
    # Get model params
    g = c.model.gravity[3]
    e3 = @SVector [0,0,1.]
    mass = c.model.mass
    J = c.model.J

    # Get time step
    k = get_k(c, t)
    b1d = c.bref[k]
    xd = c.Xref[k]

    # Parse the state
    r,q,v,Ω = Dynamics.parse_state(c.model, x)
    rd,qd,vd,Ωd = Dynamics.parse_state(c.model, xd)
    rdd,qdd,vdd,Ωdd = Dynamics.parse_state(c.model, c.Xdref[k])
    R = rotmat(q)

    # Calculate the linear errors
    ex = r-rd
    ev = v-vd
    xdd = rdd
    Ωdotd = Ωdd

    # Calculate desired attitude
    a = -c.kx*ex - c.kv*ev - mass*g*e3 + mass*xdd
    b3 = normalize(a)
    b2 = normalize(b3 × b1d)
    b1 = b2 × b3
    Rd = @SMatrix [b1[1] b2[1] b3[1];
                   b1[2] b2[2] b3[2];
                   b1[3] b2[3] b3[3]]

    # Calculate the attitude errors
    eR = 0.5*vee(Rd'R - R'Rd)
    eΩ = Ω - R'Rd*Ωd

    # Desired Forces and moments
    f = -a'R*e3
    M = -c.kR*eR - c.kΩ*eΩ + Ω × (J*Ω) - J*(skew(Ω)*R'Rd*Ωd - R'Rd*Ωdotd)

    # Convert to motor thrusts
    C = Dynamics.forceMatrix(c.model)
    u = C\(@SVector [f, M[1], M[2], M[3]])
end




############################################################################################
############################################################################################

function linearize(model::AbstractModel, xeq, ueq, dt)
    # Linearize the system about the given point
    z = KnotPoint(xeq, ueq, dt)
    ∇f = discrete_jacobian(model, z)
    ix,iu = z._x, z._u
    A = ∇f[ix,ix]
    B = ∇f[ix,iu]
    return A,B
end

function linearize(model::AbstractModel, Z::Traj)
    N = length(Z)
    n,m = size(model)
    n̄ = TO.state_diff_size(model)
    A = [@SMatrix zeros(n̄,n̄) for k = 1:N]
    B = [@SMatrix zeros(n̄,m) for k = 1:N]
    for k = 1:N
        ix,iu = Z[k]._x, Z[k]._u
        ∇f = discrete_jacobian(RK3, model, Z[k])
        x2 = discrete_dynamics(RK3, model, Z[k])
        G1 = TO.state_diff_jacobian(model, state(Z[k]))
        G2 = TO.state_diff_jacobian(model, x2)
        A[k] = G2'∇f[ix,ix]*G1
        B[k] = G2'∇f[ix,iu]
    end
    return A,B
end

function calc_LQR_gain(A::AbstractMatrix, B::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
        max_iters=100, tol=1e-4)
    P_ = similar_type(A)
    P = P_(A)
    for k = 1:max_iters
        P_ = Q + A'P*A - A'P*B*((R + B'P*B)\(B'P*A))
        err = norm(P-P_)
        P = copy(P_)
        if err < tol
            break
        end
    end
    K = -(R + B'P*B)\(B'P*A)
end

simulate(model::RigidBody, cntrl, x0::RBState, tf; kwargs...) =
    simulate(model, cntrl, build_state(model, x0), tf; kwargs...)

function simulate(model::AbstractModel, cntrl, x0, tf;
        dt=1e-4, w=1e-2)
    N = Int(round(tf/dt)) + 1
    dt = tf/(N-1)
    x = copy(x0)
    t = 0.0

    n,m = size(model)

    # Allocate storage for the state trajectory
    X = [@SVector zeros(n) for k = 1:N]

    for k = 1:N
        # Get control from the controller
        u = get_control(cntrl, x, t)

        # Add disturbances
        u += randn(m)*w

        # Simulate the system forward
        x = discrete_dynamics(RK3, model, x, u, t, dt)
        t += dt  # advance time

        # Store info
        X[k] = x
    end
    return X
end

function test_ICs(model::AbstractModel, cntrl::AbstractController, ICs::Vector{<:RBState};
        tf=10., dt=1e-4, xref=(@SVector zeros(N))) where N
    L = length(ICs)
    Xf = deepcopy(ICs)
    data = Dict{Symbol,Vector}(:Xf=>deepcopy(ICs),
        :max_err=>zeros(L), :avg_err=>zeros(L), :term_err=>zeros(L))
    for i = 1:L
        x0 = Dynamics.build_state(model, ICs[i])
        X = simulate(model, cntrl, x0, tf, dt=dt, w=0.0)
        err = map(X) do x
            dx = Dynamics.state_diff(model, x, x0)
            norm(dx)
        end
        data[:max_err][i] = maximum(err)
        data[:avg_err][i] = mean(err)
        data[:term_err][i] = err[end]
        data[:Xf][i] = RBState(model, X[end])
    end
    data
end
