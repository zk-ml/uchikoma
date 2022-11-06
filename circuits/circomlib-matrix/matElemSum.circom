pragma circom 2.0.3;
include "../Arithmetic.circom";

// sum of all elements in a matrix
template matElemSum (m,n) {
    signal input a[m][n];
    signal output out;

    signal sum[m*n];
    sum[0] <== a[0][0];
    var idx = 0;
    component add[m*n];
    
    for (var i=0; i < m; i++) {
        for (var j=0; j < n; j++) {
            if (idx > 0) {
                add[idx] = Add(1);
                add[idx].a[0] <== sum[idx-1];
                add[idx].b[0] <== a[i][j];
                // sum[idx] <== sum[idx-1] + a[i][j];
                sum[idx] <== add[idx].out[0];
            }
            idx++;
        }
    }
    out <== sum[m*n-1];
}
