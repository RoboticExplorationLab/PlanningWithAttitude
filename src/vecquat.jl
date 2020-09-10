struct VecQuat{T} <: Rotation{3,T}
    w::T
    x::T
    y::T
    z::T

    @inline function VecQuat{T}(w, x, y, z, normalize::Bool = true) where T
        if normalize
            inorm = inv(sqrt(w*w + x*x + y*y + z*z))
            new{T}(w*inorm, x*inorm, y*inorm, z*inorm)
        else
            new{T}(w, x, y, z)
        end
    end

    VecQuat{T}(q::VecQuat) where T = new{T}(q.w, q.x, q.y, q.z)
end

# ~~~~~~~~~~~~~~~ Constructors ~~~~~~~~~~~~~~~ #
# Use default map
function VecQuat(w,x,y,z, normalize::Bool = true)
    types = promote(w,x,y,z)
    VecQuat{eltype(types)}(w,x,y,z, normalize)
end

# Pass in Vectors
@inline function (::Type{Q})(q::AbstractVector, normalize::Bool = true) where Q <: VecQuat
    check_length(q, 4)
    Q(q[1], q[2], q[3], q[4], normalize)
end
@inline (::Type{Q})(q::StaticVector{4}, normalize::Bool = true) where Q <: VecQuat =
    Q(q[1], q[2], q[3], q[4], normalize)

# Copy constructors
VecQuat(q::VecQuat) = q

# Convert from UnitQuaternion
(::Type{Q})(q::UnitQuaternion) where Q <: VecQuat = Q(q.w, q.x, q.y, q.z)
(::Type{Q})(q::Vecquat) where Q <: UnitQuatenrion = Q(q.w, q.x, q.y, q.z)

# ~~~~~~~~~~~~~~~ StaticArrays Interface ~~~~~~~~~~~~~~~ #
function (::Type{Q})(t::NTuple{9}) where Q<:VecQuat
    #=
    This function solves the system of equations in Section 3.1
    of https://arxiv.org/pdf/math/0701759.pdf. This cheap method
    only works for matrices that are already orthonormal (orthogonal
    and unit length columns). The nearest orthonormal matrix can
    be found by solving Wahba's problem:
    https://en.wikipedia.org/wiki/Wahba%27s_problem as shown below.

    not_orthogonal = randn(3,3)
    u,s,v = svd(not_orthogonal)
    is_orthogoral = u * diagm([1, 1, sign(det(u * transpose(v)))]) * transpose(v)
    =#

    a = 1 + t[1] + t[5] + t[9]
    b = 1 + t[1] - t[5] - t[9]
    c = 1 - t[1] + t[5] - t[9]
    d = 1 - t[1] - t[5] + t[9]
    max_abcd = max(a, b, c, d)
    if a == max_abcd
        b = t[6] - t[8]
        c = t[7] - t[3]
        d = t[2] - t[4]
    elseif b == max_abcd
        a = t[6] - t[8]
        c = t[2] + t[4]
        d = t[7] + t[3]
    elseif c == max_abcd
        a = t[7] - t[3]
        b = t[2] + t[4]
        d = t[6] + t[8]
    else
        a = t[2] - t[4]
        b = t[7] + t[3]
        c = t[6] + t[8]
    end
    return Q(a, b, c, d)
end


function Base.getindex(q::VecQuat, i::Int)
    if i == 1
        ww = (q.w * q.w)
        xx = (q.x * q.x)
        yy = (q.y * q.y)
        zz = (q.z * q.z)

        ww + xx - yy - zz
    elseif i == 2
        xy = (q.x * q.y)
        zw = (q.w * q.z)

        2 * (xy + zw)
    elseif i == 3
        xz = (q.x * q.z)
        yw = (q.y * q.w)

        2 * (xz - yw)
    elseif i == 4
        xy = (q.x * q.y)
        zw = (q.w * q.z)

        2 * (xy - zw)
    elseif i == 5
        ww = (q.w * q.w)
        xx = (q.x * q.x)
        yy = (q.y * q.y)
        zz = (q.z * q.z)

        ww - xx + yy - zz
    elseif i == 6
        yz = (q.y * q.z)
        xw = (q.w * q.x)

        2 * (yz + xw)
    elseif i == 7
        xz = (q.x * q.z)
        yw = (q.y * q.w)

        2 * (xz + yw)
    elseif i == 8
        yz = (q.y * q.z)
        xw = (q.w * q.x)

        2 * (yz - xw)
    elseif i == 9
        ww = (q.w * q.w)
        xx = (q.x * q.x)
        yy = (q.y * q.y)
        zz = (q.z * q.z)

        ww - xx - yy + zz
    else
        throw(BoundsError(r,i))
    end
end

function Base.Tuple(q::VecQuat)
    ww = (q.w * q.w)
    xx = (q.x * q.x)
    yy = (q.y * q.y)
    zz = (q.z * q.z)
    xy = (q.x * q.y)
    zw = (q.w * q.z)
    xz = (q.x * q.z)
    yw = (q.y * q.w)
    yz = (q.y * q.z)
    xw = (q.w * q.x)

    # initialize rotation part
    return (ww + xx - yy - zz,
            2 * (xy + zw),
            2 * (xz - yw),
            2 * (xy - zw),
            ww - xx + yy - zz,
            2 * (yz + xw),
            2 * (xz + yw),
            2 * (yz - xw),
            ww - xx - yy + zz)
end

# ~~~~~~~~~~~~~~~ Getters ~~~~~~~~~~~~~~~ #
@inline scalar(q::VecQuat) = q.w
@inline vector(q::VecQuat) = SVector{3}(q.x, q.y, q.z)

"""
    params(R::Rotation)

Return an `SVector` of the underlying parameters used by the rotation representation.

# Example
```julia
p = MRP(1.0, 2.0, 3.0)
Rotations.params(p) == @SVector [1.0, 2.0, 3.0]  # true
```
"""
@inline params(q::VecQuat) = SVector{4}(q.w, q.x, q.y, q.z)

# ~~~~~~~~~~~~~~~ Initializers ~~~~~~~~~~~~~~~ #
Base.rand(::Type{<:VecQuat{T}}) where T =
    normalize(VecQuat{T}(randn(T), randn(T), randn(T), randn(T)))
Base.rand(::Type{VecQuat}) = Base.rand(VecQuat{Float64})
@inline Base.zero(::Type{Q}) where Q <: VecQuat = Q(1.0, 0.0, 0.0, 0.0)
@inline Base.one(::Type{Q}) where Q <: VecQuat = Q(1.0, 0.0, 0.0, 0.0)

# ~~~~~~~~~~~~~~~ Math Operations ~~~~~~~~~~~~~~~ #

# Inverses
conj(q::Q) where Q <: VecQuat = Q(q.w, -q.x, -q.y, -q.z, false)
inv(q::VecQuat) = conj(q)
(-)(q::Q) where Q <: VecQuat = Q(-q.w, -q.x, -q.y, -q.z, false)

# Norms
LinearAlgebra.norm(q::VecQuat) = sqrt(q.w^2 + q.x^2 + q.y^2 + q.z^2)
vecnorm(q::VecQuat) = sqrt(q.x^2 + q.y^2 + q.z^2)

function LinearAlgebra.normalize(q::Q) where Q <: VecQuat
    n = inv(norm(q))
    Q(q.w*n, q.x*n, q.y*n, q.z*n)
end

# Identity
(::Type{Q})(I::UniformScaling) where Q <: VecQuat = one(Q)

# Exponentials and Logarithms
"""
    pure_quaternion(v::AbstractVector)
    pure_quaternion(x, y, z)

Create a `VecQuat` with zero scalar part (i.e. `q.w == 0`).
"""
function pure_quaternion(v::AbstractVector)
    check_length(v, 3)
    VecQuat(zero(eltype(v)), v[1], v[2], v[3], false)
end

@inline pure_quaternion(x::Real, y::Real, z::Real) =
    VecQuat(zero(x), x, y, z, false)

function exp(q::Q) where Q <: VecQuat
    θ = vecnorm(q)
    sθ,cθ = sincos(θ)
    es = exp(q.w)
    M = es*sθ/θ
    Q(es*cθ, q.x*M, q.y*M, q.z*M, false)
end

function expm(ϕ::AbstractVector)
    check_length(ϕ, 3)
    θ = norm(ϕ)
    sθ,cθ = sincos(θ/2)
    M = 1//2 *sinc(θ/π/2)
    VecQuat(cθ, ϕ[1]*M, ϕ[2]*M, ϕ[3]*M, false)
end

function log(q::Q, eps=1e-6) where Q <: VecQuat
    # Assumes unit quaternion
    θ = vecnorm(q)
    if θ > eps
        M = atan(θ, q.w)/θ
    else
        M = (1-(θ^2/(3q.w^2)))/q.w
    end
    pure_quaternion(M*vector(q))
end

function logm(q::VecQuat)
    # Assumes unit quaternion
    2*vector(log(q))
end

# Composition
"""
    (*)(q::VecQuat, w::VecQuat)

Quternion Composition

Equivalent to
```julia
lmult(q) * SVector(w)
rmult(w) * SVector(q)
```

Sets the output mapping equal to the mapping of `w`
"""
function (*)(q::VecQuat, w::VecQuat)
    VecQuat(q.w * w.w - q.x * w.x - q.y * w.y - q.z * w.z,
                   q.w * w.x + q.x * w.w + q.y * w.z - q.z * w.y,
                   q.w * w.y - q.x * w.z + q.y * w.w + q.z * w.x,
                   q.w * w.z + q.x * w.y - q.y * w.x + q.z * w.w, false)
end

"""
    (*)(q::VecQuat, r::StaticVector)

Rotate a vector

Equivalent to `hmat()' lmult(q) * rmult(q)' hmat() * r`
"""
function Base.:*(q::VecQuat, r::StaticVector)  # must be StaticVector to avoid ambiguity
    check_length(r, 3)
    w = q.w
    v = vector(q)
    (w^2 - v'v)*r + 2*v*(v'r) + 2*w*cross(v,r)
end

"""
    (*)(q::VecQuat, w::Real)

Scalar multiplication of a quaternion. Breaks unit norm.
"""
function (*)(q::Q, w::Real) where Q<:VecQuat
    return Q(q.w*w, q.x*w, q.y*w, q.z*w, false)
end
(*)(w::Real, q::VecQuat) = q*w



(\)(q1::VecQuat, q2::VecQuat) = conj(q1)*q2  # Equivalent to inv(q1)*q2
(/)(q1::VecQuat, q2::VecQuat) = q1*conj(q2)  # Equivalent to q1*inv(q2)

(\)(q::VecQuat, r::SVector{3}) = conj(q)*r          # Equivalent to inv(q)*r

"""
    rotation_between(from, to)

Compute the quaternion that rotates vector `from` so that it aligns with vector
`to`, along the geodesic (shortest path).
"""
rotation_between(from::AbstractVector, to::AbstractVector) = rotation_between(SVector{3}(from), SVector{3}(to))
function rotation_between(from::SVector{3}, to::SVector{3})
    # Robustified version of implementation from https://www.gamedev.net/topic/429507-finding-the-quaternion-betwee-two-vectors/#entry3856228
    normprod = sqrt(dot(from, from) * dot(to, to))
    T = typeof(normprod)
    normprod < eps(T) && throw(ArgumentError("Input vectors must be nonzero."))
    w = normprod + dot(from, to)
    v = abs(w) < 100 * eps(T) ? perpendicular_vector(from) : cross(from, to)
    @inbounds return VecQuat(w, v[1], v[2], v[3]) # relies on normalization in constructor
end



# ~~~~~~~~~~~~~~~~~~~ Rotations Stuff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
function kinematics(q::Q, ω::AbstractVector) where Q <: VecQuat 
    1//2 * params(q*Q(0.0, ω[1], ω[2], ω[3], false))
end

function ∇differential(q::VecQuat)
    I(4)
end

function ∇²differential(q::VecQuat, b::AbstractVector)
    @SMatrix zeros(4,4)
end