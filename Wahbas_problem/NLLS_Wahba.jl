# activate the virtual environment
# cd(dirname(@__FILE__))
import Pkg; Pkg.activate(@__DIR__)

using ForwardDiff, LinearAlgebra, Statistics, JLD2
# using MATLAB

function randq()
    """Random quaternion"""
    return normalize(randn(4))
end

function so3noise(v,std)
    """Apply 3d gaussian noise on S03"""
    noise_phi = std*randn(3)
    return exp(skew(noise_phi))*v
end

function angle_error(q1,q2)
    """Get angle error between two attitudes"""
    q = qconj(q1) ⊙ q2
    v = q[2:4]
    s = q[1]
    normv = norm(v)
    θ = (2 * atan(normv, s))
    return rad2deg(θ)
end

function p_from_q(q)
    """Rodgrigues parameter from quaternion"""
    s = q[1]
    v = q[2:4]
    return v/s
end

function q_from_p(p)
    """quaternion from rodrigues parameter"""
    return (1/(sqrt(1 + p'*p)))*[1;p]
end

function qconj(q)
    """quaternion conjugate"""
    s = q[1]
    v = q[2:4]
    return [s;-v]
end

function q_shorter(q)
    """Get the quaternion that represents θ < π """
    if q[1]<0
        q = -q
    end
    return q
end

function dcm_from_q(q)
    """DCM from quaternion"""
    s = q[1]
    v = q[2:4]
    return I + 2*skew(v)*(s*I + skew(v))
end
function q_from_phi(phi)
    """Quaternion from axis angle"""
    θ = norm(phi)
    r = phi/θ
    return [cos(θ/2);r*sin(θ/2)]
end

function ⊙(q1, q2)
    """Hamilton product, quaternion multiplication"""
    v1 = q1[2:4]
    s1 = q1[1]
    v2 = q2[2:4]
    s2 = q2[1]
    return [(s1 * s2 - dot(v1, v2)); (s1 * v2 + s2 * v1 + cross(v1, v2))]
end

function skew(v)
    """Skew-symmetric matrix from 3 element vector"""
    return [0 -v[3] v[2]; v[3] 0 -v[1]; -v[2] v[1] 0]
end

# function L(q)
#     qs = q[1]
#     qv = q[2:4]
#     return [qs  -qv'; qv (qs*I + skew(qv))]
# end
#
#
# function R(q)
#     qs = q[1]
#     qv = q[2:4]
#     return [qs  -qv'; qv (qs*I - skew(qv))]
# end
#
# function H()
#     return [zeros(1,3);diagm(ones(3))]
# end

function G(q)
    """Quaternion to rodrigues parameter Jacobian"""
    s = q[1]
    v = q[2:4]
    return [-v'; (s*I + skew(v))]
end

function generate_data(std,N)
    """Generate the data for Wahba's problem"""
    # create random attitude
    ᴺqᴮ = randq()
    ᴺQᴮ = dcm_from_q(ᴺqᴮ)

    # equal weights
    w = ones(N)/N

    # create true and measured vectors
    Rn = fill(zeros(3),N)
    Rb = fill(zeros(3),N)
    for i = 1:N
        Rn[i] = normalize(randn(3))
        Rb[i] = transpose(ᴺQᴮ)*so3noise(Rn[i],deg2rad(std))
    end

    return w, Rn, Rb, ᴺqᴮ, ᴺQᴮ
end
function q_from_dcm(dcm)
    """Kane/Levinson convention, scalar first"""
    R = dcm
    T = R[1,1] + R[2,2] + R[3,3]
    if T> R[1,1] && T > R[2,2] && T>R[3,3]
        q4 = .5*sqrt(1+T)
        r  = .25/q4
        q1 = (R[3,2] - R[2,3])*r
        q2 = (R[1,3] - R[3,1])*r
        q3 = (R[2,1] - R[1,2])*r
    elseif R[1,1]>R[2,2] && R[1,1]>R[3,3]
        q1 = .5*sqrt(1-T + 2*R[1,1])
        r  = .25/q1
        q4 = (R[3,2] - R[2,3])*r
        q2 = (R[1,2] + R[2,1])*r
        q3 = (R[1,3] + R[3,1])*r
    elseif R[2,2]>R[3,3]
        q2 = .5*sqrt(1-T + 2*R[2,2])
        r  = .25/q2
        q4 = (R[1,3] - R[3,1])*r
        q1 = (R[1,2] + R[2,1])*r
        q3 = (R[2,3] + R[3,2])*r
    else
        q3 = .5*sqrt(1-T + 2*R[3,3])
        r  = .25/q3
        q4 = (R[2,1] - R[1,2])*r
        q1 = (R[1,3] + R[3,1])*r
        q2 = (R[2,3] + R[3,2])*r
    end
    q = [q4;q1;q2;q3]
    return q
end

function svd_wahba(Rn,Rb,w)
    """Wahba's problem SVD solution"""

    # attitude profile matrix
    B = zeros(3,3)
    for i = 1:length(Rn)
        B = B +  w[i]*Rb[i]*Rn[i]'
    end

    # solve WAHBAHs problem using SVD
    F = svd(B')

    # DCM for SVD solution
    Qsvd = F.U*diagm([1; 1; det(F.U)*det(F.V)])*F.V'

    # quaternion for SVD solution
    qsvd = q_from_dcm(Qsvd)

    return Qsvd, qsvd
end

function r_fx(q,Rn,Rb,w)
    """returns r where Wahba's cost function L is rᵀr"""

    # allocate in a fwd diff friendly way
    r = zeros(eltype(q),length(Rn)*3)

    # stack the residuals
    for i = 1:length(Rn)
        r[((i-1)*3 + 1):((i-1)*3 + 3)] = sqrt(w[i])*(Rn[i] - dcm_from_q(q)*Rb[i])
    end

    return r
end


function S(q,Rn,Rb,w)
    """sum of squared residuals"""
    r = r_fx(q,Rn,Rb,w)
    return r'*r
end

function newton_wahba()
    """Solve Wahba's problem using a Gauss-Newton NLLS method"""

    # generate 4 vector measurements
    w, Rn, Rb, ᴺqᴮ, ᴺQᴮ = generate_data(5,20)

    # solve Wahba's problem SVD style
    Qsvd, qsvd = svd_wahba(Rn,Rb,w)

    # jacobian function for the residuals
    J_fx = q -> ForwardDiff.jacobian(q -> r_fx(q,Rn,Rb,w),q)

    # here we print the cost attained by the SVD solution
    SVD_cost = S(qsvd,Rn,Rb,w)

    # set initial guess to q
    q = qsvd ⊙ q_from_phi(deg2rad(10)*normalize(randn(3)))

    err_hist = NaN*ones(21)

    # Gauss-Newton
    for ii = 1:4

        # jacobian (we use G to get the rodrigues parameter version)
        J = J_fx(q)*G(q)

        # step direction
        v = -J\r_fx(q,Rn,Rb,w)

        # current cost value
        S_k = S(q,Rn,Rb,w)
        α = 1.0
        dS = 0.0
        S_new = 0.0

        # print first cost difference
        if ii==1
            err_hist[ii] = angle_error(q,qsvd)

            println("")
            println("Iteration:   ",ii-1,"        α:      ",α,"    ΔJ:     ",S_k-SVD_cost)
        end

        # line search
        for jj = 1:20

            # take step (apply step multiplicatively)
            qnew = q ⊙ q_from_p(α*v)

            # get new cost value
            S_new = S(qnew,Rn,Rb,w)

            if S_new<S_k # if it's better, we keep it
                q = copy(qnew)
                dS = abs(S_new-S_k)

                break
            else         # if it's worse, we backtrack
                α *= .5
            end
        end

        err_hist[ii+1] = angle_error(q,qsvd)

        println("Iteration:   ",ii,"        α:      ",α,"    ΔJ:     ",S_new-SVD_cost)

        if err_hist[ii+1]<1e-8
            break
        end
    end


    return q, qsvd, err_hist


end
function projected_newton_wahba()
    """Solve Wahba's problem using a Gauss-Newton NLLS method"""

    # generate 4 vector measurements
    w, Rn, Rb, ᴺqᴮ, ᴺQᴮ = generate_data(5,20)

    # solve Wahba's problem SVD style
    Qsvd, qsvd = svd_wahba(Rn,Rb,w)

    # jacobian function for the residuals
    J_fx = q -> ForwardDiff.jacobian(q -> r_fx(q,Rn,Rb,w),q)

    # here we print the cost attained by the SVD solution
    SVD_cost = S(qsvd,Rn,Rb,w)

    # set initial guess to q from triad
    q = qsvd ⊙ q_from_phi(deg2rad(10)*normalize(randn(3)))

    err_hist = NaN*ones(21)

    # Gauss-Newton
    for ii = 1:4

        # jacobian (we use G to get the rodrigues parameter version)
        J = J_fx(q)

        # step direction
        v = -J\r_fx(q,Rn,Rb,w)

        # current cost value
        S_k = S(q,Rn,Rb,w)
        α = 1.0
        dS = 0.0
        S_new = 0.0

        # print first cost difference
        if ii==1
            err_hist[ii] = angle_error(q,qsvd)
            println("")
            println("Iteration:   ",ii-1,"        α:      ",α,"    ΔJ:     ",S_k-SVD_cost)
        end

        # line search
        for jj = 1:200

            # take step (apply step multiplicatively)
            qnew = normalize(q + (α*v))

            # get new cost value
            S_new = S(qnew,Rn,Rb,w)

            if S_new<S_k # if it's better, we keep it
                q = copy(qnew)
                dS = abs(S_new-S_k)

                break
            else         # if it's worse, we backtrack
                α *= .5
            end
        end

        err_hist[ii+1] = angle_error(q,qsvd)
        println("Iteration:   ",ii,"        α:      ",α,"    ΔJ:     ",S_new-SVD_cost)

        if err_hist[ii+1]<1e-6
            break
        end
    end


    return q, qsvd, err_hist


end

## run trials
trials = 25
max_iters = 21
mn = zeros(trials,max_iters)
pn = zeros(trials,max_iters)

for i = 1:trials
    q, qsvd, mn[i,:] = newton_wahba()
    q, qsvd, pn[i,:] = projected_newton_wahba()
end
@save joinpath(@__DIR__, "wahba_results.jld2") mn pn



## Generate plot
using PGFPlotsX, Statistics
@load joinpath(@__DIR__, "wahba_results.jld2") mn pn

# average the results
p_average = zeros(5)
m_average = zeros(5)

for i = 1:5
    p_average[i] = (mean(pn[:,i]))
    m_average[i] = (mean(mn[:,i]))
end

p_plots = map(eachrow(pn)) do p
    @pgf PlotInc(
        {
            color="cyan",
            no_marks,
            "line width=0.1pt",
            "forget plot",
            opacity=0.3
            # "dashed"
        },
        Coordinates(1:5, p[1:5])
    )
end
m_plots = map(eachrow(mn)) do p
    @pgf PlotInc(
        {
            color="orange",
            no_marks,
            "line width=0.1pt",
            # "dashed",
            "forget plot",
            opacity=0.3
        },
        Coordinates(1:5, p[1:5])
    )
end
p = @pgf Axis(
    {
        xlabel="iterations",
        ylabel="angle error (degrees)",
        "ymode=log",
        xmajorgrids,
        ymajorgrids,
        "legend style={at={(0.1,0.1)},anchor=south west}"
    },
    m_plots...,
    p_plots...,
    PlotInc(
        {
            color="cyan",
            no_marks,
            "line width=3pt"
        },
        Coordinates(1:5, p_average)
    ),
    PlotInc(
        {
            color="orange",
            no_marks,
            "line width=3pt"
        },
        Coordinates(1:5, m_average)
    ),
    Legend("naive", "modified")
)
pgfsave("paper/figures/wahba_convergence.tikz", p, include_preamble=false)

mat"
c = 0.7
figure
hold on
title('Solving Wahba''s Problem with Newton''s Method')
plot(0:20,$pn','color',[c c 1]);
plot(0:20,$mn','color',[1 c c]);
p1 = plot(0:4,$p_average,'b','linewidth',2);
p2 = plot(0:4,$m_average,'r','linewidth',2);
set(gca,'yscale','log')
xlim([0,4])
ylabel('Degree Error')
xlabel('Iterations')
%legend('Projected Newton','Multiplicative Newton')
legend([p1(1),p2(1)],'Projected Newton','Multiplicative Newton',...
'Location','SouthWest')
hold off
%matlab2tikz('convergence_plot.tex')
"
