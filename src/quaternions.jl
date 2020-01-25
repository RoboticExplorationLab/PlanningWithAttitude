export
    cayley_map,
    cayley_jacobian,
    Lmult,
    Rmult,
    Vmat,
    Tmat

function Lmult(q::SVector{4})
    @SMatrix [q[1] -q[2] -q[3] -q[4];
              q[2]  q[1] -q[4]  q[3];
              q[3]  q[4]  q[1] -q[2];
              q[4] -q[3]  q[2]  q[1]]
end

function Rmult(q::SVector{4})
    @SMatrix [q[1] -q[2] -q[3] -q[4];
              q[2]  q[1]  q[4] -q[3];
              q[3] -q[4]  q[1]  q[2];
              q[4]  q[3] -q[2]  q[1]]
end

function Vmat()
    @SMatrix [
        0 1 0 0;
        0 0 1 0;
        0 0 0 1.
    ]
end

function Tmat()
    @SMatrix [
        1  0  0  0;
        0 -1  0  0;
        0  0 -1  0;
        0  0  0 -1;
    ]
end

function cayley_map(g::SVector{3})
    M = 1/sqrt(1+g'g)
    @SVector [M, M*g[1], M*g[2], M*g[3]]
end

function cayley_jacobian(g::SVector{3})
    n = 1+g'g
    ni = 1/n
    [-g'; -g*g' + I*n]*ni*sqrt(ni)
end
