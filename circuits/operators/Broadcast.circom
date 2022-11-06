pragma circom 2.1.0;

template Broadcast3DAxis0Sub (i1, i2, i3) {
    signal input A[i1][i2][i3];
    signal input B[i1][1][1];
    signal output out[i1][i2][i3];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== A[i][j][k] - B[i][0][0];
            }
        }
    }
}

template Broadcast3DAxis0Add (i1, i2, i3) {
    signal input A[i1][i2][i3];
    signal input B[i1][1][1];
    signal output out[i1][i2][i3];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2; j++) {
            for (var k = 0; k < i3; k++) {
                out[i][j][k] <== A[i][j][k] + B[i][0][0];
            }
        }
    }
}
