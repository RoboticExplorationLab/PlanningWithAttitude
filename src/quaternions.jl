
function Lmult(q::SVector{4})
    @SMatrix [q[1] -q[2] -q[3] -q[4];
              q[2]  q[1] -q[4]  q[3];
              q[3]  q[4]  q[1] -q[2];
              q[4] -q[3]  q[2]  q[1]]
end

function Vmat()
    @SMatrix [
        0 1 0 0;
        0 0 1 0;
        0 0 0 1.
    ]
end
