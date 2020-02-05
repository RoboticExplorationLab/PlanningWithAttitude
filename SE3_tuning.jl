# Test SE(3) controller
Rot = UnitQuaternion{Float64,CayleyMap}
# lee = Dynamics.LeeQuad(Rot)
model = Dynamics.Quadrotor2{Rot}()
# eigen(lee.J)
# r = maximum(eval(model.J)) / maximum(eval(lee.J))
# kx = 59.02 * r + 1
# kv = 24.3  * r + 0.3
# kR = 8.8   * r + 2
# kO = 1.54  * r + 0.05
kx = 2.71 + 1.0
kv = 1.01 + 0.5
kR = 2.26*Diagonal(@SVector [1,1,1.])
kO = 0.3*Diagonal(@SVector [1,1,1.])
x0 = zeros(model)[1]
u0 = Dynamics.trim_controls(model)

tf = 10.0
dt = 1e-4
times = range(0,tf,step=dt)
Xref = [copy(x0) for k = 1:length(times)]
Xdref = [copy(x0) for k = 1:length(times)]
bref = [@SVector [1,0,0.] for k = 1:length(times)]

se3 = SE3Tracking(model, Xref, Xdref, bref, collect(times), kx=kx, kv=kv, kR=kR[1], kO=kO[1])
cntrl = PlanningWithAttitude.HFCA(model, Xref, Xdref, bref, collect(times), kx=kx, kv=kv, kR=kR, kO=kO)
get_control(cntrl, x0, 0.0)
get_control(se3, x0, 0.0)

u = normalize(@SVector randn(3))
# u = @SVector [1,0,0.]
qinit = expm(u * deg2rad(170))
xinit = Dynamics.build_state(model, [0,0.2,0], qinit, randn(3)*0.1, randn(3)*0.2)
X = simulate(model, cntrl, xinit, tf)
visualize!(vis, model, X[1:100:length(X)], tf)
