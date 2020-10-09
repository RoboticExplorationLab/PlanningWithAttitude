struct QuatRenorm{R,L} <: RD.RigidBody{R}
    model::L
    function QuatRenorm(model::RD.RigidBody{R}) where R
        new{R,typeof(model)}(model)
    end
end
RD.control_dim(model::QuatRenorm) = RD.control_dim(model.model)

function RD.discrete_dynamics(::Type{Q}, 
        model::QuatRenorm, x::StaticVector, u::StaticVector, t, dt) where Q
    x2 = RD.discrete_dynamics(Q, model.model, x, u, t, dt)
    # renormalize the quaternion
    r,q,v,ω = RobotDynamics.parse_state(model.model, x2, true)
    RobotDynamics.build_state(model.model, r,q,v,ω)
end
# RD.wrenches(model::QuatRenorm, z::RD.AbstractKnotPoint) = RD.wrenches(model.model, z)
# RD.forces(model::QuatRenorm, x, u) = RD.forces(model.model, x, u)
# RD.moments(model::QuatRenorm, x::StaticVector, u::StaticVector) = RD.moments(model.model, x, u)
# RD.mass(model::QuatRenorm) = RD.mass(model.model)
# RD.inertia(model::QuatRenorm) = RD.inertia(model.model)
# RD.inertia_inv(model::QuatRenorm) = RD.inertia_inv(model.model)
# RD.velocity_frame(model::QuatRenorm) = RD.velocity_frame(model.model)



struct QuatSlackModel{R,M,L} <: RD.RigidBody{R} 
    model::L
    function QuatSlackModel(model::L) where L <: RD.RigidBody
        M = RD.control_dim(model)
        R = RD.rotation_type(model)
        new{R,M,L}(model)
    end
end
RobotDynamics.state_dim(model::QuatSlackModel) = RobotDynamics.state_dim(model.model)
RobotDynamics.control_dim(::QuatSlackModel{<:Any,M}) where M = M+1 

function RobotDynamics.discrete_dynamics(::Type{Q}, model::QuatSlackModel, x, u, t, dt) where {Q}
    s = u[end]                       # quaternion slack
    u0 = pop(u)                      # original controls
    x2 = RD.discrete_dynamics(Q, model.model, x, u, t, dt)
    r,q,v,ω = RobotDynamics.parse_state(model, x2)
    q̄ = (1-s)*q                      # scale the quaternion by the extra control
    x̄ = RobotDynamics.build_state(model, r, q̄, v, ω)
end

"""
    UnitQuatConstraint

Enforce a that the unit quaternion in the state, at indices `qind`, has unit norm
after being multiplied by a slack variable at index `sind` in the joint state-control vector
`z = [x;u]`.
"""
struct UnitQuatConstraint <: TrajectoryOptimization.StateConstraint
    n::Int
    qind::SVector{4,Int}
end

function UnitQuatConstraint(n::Int, qind=4:7)
    UnitQuatConstraint(n, SVector{4}(qind))
end

UnitQuatConstraint(model::QuatSlackModel) = UnitQuatConstraint(state_dim(model)) 

Base.copy(con::UnitQuatConstraint) = UnitQuatConstraint(con.n, con.qind)

@inline TO.sense(::UnitQuatConstraint) = TO.Equality()
@inline Base.length(::UnitQuatConstraint) = 1
@inline RobotDynamics.state_dim(con::UnitQuatConstraint) = con.n

function TO.evaluate(con::UnitQuatConstraint, z::RD.AbstractKnotPoint)
    q = z.z[con.qind]
    return SA[q'q - 1]
end

function TO.jacobian!(∇c, con::UnitQuatConstraint, z::RD.AbstractKnotPoint)
    q = z.z[con.qind]
    for (i,j) in enumerate(con.qind)
        ∇c[1,j] = 2*q[i]
    end
    # ∇c[con.sind] = 2s*dot(q,q)
    return false
end

function TO.change_dimension(con::UnitQuatConstraint, n::Int, m::Int, xi=1:n, ui=1:m)
    UnitQuatConstraint(n, xi[con.qind])
end