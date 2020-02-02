
# Test SE(3) controller
Rot = UnitQuaternion{Float64,CayleyMap}
lee = Dynamics.LeeQuad(Rot)
model = Dynamics.Quadrotor2{Rot}()
eigen(lee.J)
r = maximum(eval(model.J)) / maximum(eval(lee.J))
kx = 59.02 * r + 1
kv = 24.3  * r + 0.3
kR = 8.8   * r + 2
kO = 1.54  * r + 0.05
x0 = zeros(model)[1]
u0 = Dynamics.trim_controls(model)

tf = 10.0
dt = 1e-4
times = range(0,tf,step=dt)
Xref = [copy(x0) for k = 1:length(times)]
bref = [@SVector [1,0,0.] for k = 1:length(times)]

cntrl = SE3Tracking(model, Xref, bref, collect(times), kx=kx, kv=kv, kR=kR, kO=kO)

u = normalize(@SVector randn(3))
qinit = expm(u * deg2rad(170))
xinit = Dynamics.build_state(model, [0,0.2,0], qinit, randn(3)*0.1, randn(3)*0.2)
X = simulate(model, cntrl, xinit, tf)
visualize!(vis, model, X[1:100:length(X)], tf)
PlanningWithAttitude.linearize(lee, x0, u0, 0.)
X
