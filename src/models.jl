export
	AbstractModel,
    Satellite,
	dynamics,
	discrete_dynamics,
	discrete_jacobian,
	dynamics_jacobian,
	orientation

abstract type AbstractModel end

@inline Base.position(::AbstractModel, x::SVector) = @SVector [x[1], x[2], x[3]]
@inline orientation(::AbstractModel, x::SVector) = @SVector [x[4], x[5], x[6], x[7]]


function discrete_dynamics(model::AbstractModel, x::SVector{N,T}, u::SVector{M,T},
		dt::T) where {N,M,T}
    k1 = dynamics(model, x, u)*dt;
    k2 = dynamics(model, x + k1/2, u)*dt;
    k3 = dynamics(model, x - k1 + 2*k2, u)*dt;
    x + (k1 + 4*k2 + k3)/6
end
@inline discrete_dynamics(model::AbstractModel, z::KnotPoint) =
	discrete_dynamics(model, state(z), control(z), z.dt)

function discrete_jacobian(model::AbstractModel,
        z::KnotPoint{T,N,M,NM}) where {T,N,M,NM}
    ix,iu,idt = z._x, z._u, NM+1
    fd_aug(s) = discrete_dynamics(model, s[ix], s[iu], s[idt])
    s = [z.z; @SVector [z.dt]]
    ForwardDiff.jacobian(fd_aug, s)
end


struct Satellite <: AbstractModel
    J::Diagonal{Float64,SVector{3,Float64}}
end

Satellite() = Satellite(Diagonal(@SVector ones(3)))

Base.size(::Satellite) = 7,3
Base.position(::Satellite, x::SVector) = @SVector zeros(3)

function dynamics(model::Satellite, x::SVector, u::SVector)
    ω = @SVector [x[1], x[2], x[3]]
    q = normalize(@SVector [x[4], x[5], x[6], x[7]])
    J = model.J

    ωdot = J\(u - ω × (J*ω))
    qdot = 0.5*Lmult(q)*Vmat()'ω
    return [ωdot; qdot]
end

function state_diff(model::Satellite, x::SVector, x0::SVector)::SVector{6}
    ω = @SVector [x[1], x[2], x[3]]
    q = @SVector [x[4], x[5], x[6], x[7]]
    ω0 = @SVector [x0[1], x0[2], x0[3]]
    q0 = @SVector [x0[4], x0[5], x0[6], x0[7]]

    δω = ω - ω0
    δq = Lmult(q0)'q
    ϕ = @SVector [δq[2]/δq[1], δq[3]/δq[1], δq[4]/δq[1]]
    return [δω; ϕ]
end

function state_diff_jacobian(model::Satellite, x::SVector)
    q = @SVector [x[4], x[5], x[6], x[7]]
    G = Lmult(q)*Vmat()'
    return @SMatrix [1 0 0 0 0 0;
                     0 1 0 0 0 0;
                     0 0 1 0 0 0;
                     0 0 0 G[1] G[5] G[ 9];
                     0 0 0 G[2] G[6] G[10];
                     0 0 0 G[3] G[7] G[11];
                     0 0 0 G[4] G[8] G[12];
                     ]
end

function Base.rand(::Satellite)
	ω = @SVector rand(3)
	q = normalize(@SVector randn(4))
	u = @SVector rand(3)
	return [ω; q], u
end
