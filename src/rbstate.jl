
struct RBState{T}
    r::SVector{3,T}
    q::UnitQuaternion{T,TO.DEFAULT_QUATDIFF}
    v::SVector{3,T}
    ω::SVector{3,T}
end

function RBState(r::AbstractVector, q::Rotation, v::AbstractVector, ω::AbstractVector)
    r_ = @SVector [r[1],r[2],r[3]]
    q_ = UnitQuaternion{TO.DEFAULT_QUATDIFF}(q)
    v_ = @SVector [v[1],v[2],v[3]]
    ω_ = @SVector [ω[1],ω[2],ω[3]]
    RBState(r_, q_, v_, ω_)
end

function RBState(x::SVector{13})
    r_ = @SVector [x[1],x[2],x[3]]
    q_ = UnitQuaternion(x[4], x[5], x[6], x[7])
    v_ = @SVector [x[8],x[9],x[10]]
    ω_ = @SVector [x[11],x[12],x[13]]
    RBState(r_, q_, v_, ω_)
end

function RBState(model::RigidBody, x::SVector)
    r,q,v,ω = Dynamics.parse_state(model, x)
    RBState(r,UnitQuaternion(q),v,ω)
end

function Dynamics.build_state(model::RigidBody{R}, rbs::RBState) where R
    Dynamics.build_state(model, rbs.r, rbs.q, rbs.v, rbs.ω)
end

function Base.:+(s1::RBState, s2::RBState)
    RBState(s1.r+s2.r, s1.q*s2.q, s1.v+s2.v, s1.ω+s2.ω)
end

function Base.:-(s1::RBState, s2::RBState)
    RBState(s1.r-s2.r, s2.q\s1.q, s1.v-s2.v, s1.ω-s2.ω)
end


Base.zero(s1::RBState) = zero(RBState)
function Base.zero(::Type{<:RBState})
    RBState((@SVector zeros(3)), UnitQuaternion(I),
        (@SVector zeros(3)), (@SVector zeros(3)))
end

function randbetween(xmin::RBState, xmax::RBState)
    dx = xmax - xmin
    RBState(
        xmin.r .+ rand(3) .* dx.r,
        # rand(UnitQuaternion),
        expm((@SVector randn(3))*rand()*deg2rad(170)),
        xmin.v .+ rand(3) .* dx.v,
        xmin.ω .+ rand(3) .* dx.ω
    )
end

function LinearAlgebra.norm(s::RBState)
    sqrt(s.r's.r + s.v's.v + s.ω's.ω + LinearAlgebra.norm2(s.q))
end
