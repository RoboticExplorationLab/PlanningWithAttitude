struct VecModel{L} <: RD.AbstractModel
    model::L
    VecModel(model::L) where L <: RD.LieGroupModel = new{L}(model)
end

RobotDynamics.state_dim(model::VecModel) = RD.state_dim(model.model)
RobotDynamics.control_dim(model::VecModel) = RD.control_dim(model.model)
RobotDynamics.dynamics(model::VecModel, args...) = RD.dynamics(model.model, args...) 