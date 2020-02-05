using MeshCat
using GeometryTypes
using Plots
using CoordinateTransformations

# vis = Visualizer()
# open(vis)

rad = 0.5f0
w,h = 1.5,1.5
point = HyperSphere(Point3f0(0,0,0), 0.01f0)
ball = HyperSphere(Point3f0(0,0,0), rad)
plane = HyperRectangle(Vec(-w/2,-h/2,rad), Vec(w,h,0.01))
s = 0.8
setobject!(vis["ball"], ball, MeshPhongMaterial(color=RGBA(0.5*s,0.5*s,1*s, 1.0)))
setobject!(vis["ball"]["plane"], plane, MeshBasicMaterial(color=RGBA(0.7,0,0,1)))
setobject!(vis["ball"]["point"], point, MeshBasicMaterial(color=RGBA(0,0,0,1)))
settransform!(vis["ball"], LinearMap(RotX(deg2rad(-30))))
settransform!(vis["ball"]["point"], Translation(0,0,rad))
