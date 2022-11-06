pragma circom 2.0.3;
include "../Arithmetic.circom";

// matrix multiplication by element
template matElemMul (m,n) {
    signal input a[m][n];
    signal input b[m][n];
    signal output out[m][n];
    
    var idx = 0;
    component mul[m*n];
    for (var i=0; i < m; i++) {
        for (var j=0; j < n; j++) {
            mul[idx] = Mul(1);
            mul[idx].a[0] <== a[i][j];
            mul[idx].b[0] <== b[i][j];
            // out[i][j] <== a[i][j] * b[i][j];
            out[i][j] <== mul[idx].out[0];
            idx++;
        }
    }
}
