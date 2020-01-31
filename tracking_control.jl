
abstract type AbstractController end
abstract type LQRController <: AbstractController end
abstract type TimeVaryingController <: AbstractController end

import TrajectoryOptimization: state_diff, state_diff_jacobian

struct TVLQR{L,N,M,TK} <: TimeVaryingController
    Xref::Vector{SVector{N,Float64}}
    K::Vector{TK}
    d::Vector{SVector{M,Float64}}
    t::Vector{Float64}
end

function get_control(cntrl::TVLQR, x, t)
    k = get_k(cntrl, t)
    dx = x - cntrl.Xref[k]
    return cntrl.K*dx + cntrl.d
end

function get_k(cntrl::TVLQR, t)
    findsortedlast(cntrl.t,t)
end

struct LQR{T,N,M,TK} <: LQRController
    K::TK
    xref::SVector{N,T}
    uref::SVector{M,T}
end

function LQR(model::AbstractModel, xeq::AbstractVector, ueq::AbstractVector, dt::Real, Q, R)
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

struct MLQR{T,N,M,TK} <: LQRController
    model::AbstractModel
    K::TK
    xref::SVector{N,T}
    uref::SVector{M,T}
end

function MLQR(model::AbstractModel, xeq, ueq, dt::Real, Q, R)
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

function calc_LQR_gain(A::AbstractMatrix, B::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix;
        max_iters=100, tol=1e-4)
    P = copy(Q)
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


function simulate(model::AbstractModel, cntrl, x0, tf; dt=1e-4)
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
        u += randn(m)*1e-1

        # Simulate the system forward
        x = discrete_dynamics(RK3, model, x, u, t, dt)
        t += dt  # advance time

        # Store info
        X[k] = x
    end
    return X
end
