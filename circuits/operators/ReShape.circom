pragma circom 2.1.0;

template ReShape2D(o1, o2) {
    signal input in[o1 * o2];
    signal output out[o1][o2];

    var idx = 0;
    for (var i=0; i < o1; i++) {
        for (var j=0; j < o2; j++) {
            out[i][j] <== in[idx];
            idx ++;
        }
    }
}

template ReShape3D(o1, o2, o3) {
    signal input in[o1 * o2 * o3];
    signal output out[o1][o2][o3];

    var idx = 0;
    for (var i=0; i < o1; i++) {
        for (var j=0; j < o2; j++) {
            for (var k=0; k < o3; k++) {
                out[i][j][k] <== in[idx];
                idx ++;
            }
        }
    }
}

template ReShape4D(o1, o2, o3, o4) {
    signal input in[o1 * o2 * o3 * o4];
    signal output out[o1][o2][o3][o4];

    var idx = 0;
    for (var i=0; i < o1; i++) {
        for (var j=0; j < o2; j++) {
            for (var k=0; k < o3; k++) {
                for (var m=0; m < o4; m++) {
                    out[i][j][k][m] <== in[idx];
                    idx ++;
                }
            }
        }
    }
}

template Flatten2D(i1, i2) {
    signal input in[i1][i2];
    signal output out[i1 * i2];

    var idx = 0;
    for (var i=0; i < i1; i++) {
        for (var j=0; j < i2; j++) {
            out[idx] <== in[i][j];
            idx ++;
        }
    }
}

template Flatten3D(i1, i2, i3) {
    signal input in[i1][i2][i3];
    signal output out[i1 * i2 * i3];

    var idx = 0;
    for (var i=0; i < i1; i++) {
        for (var j=0; j < i2; j++) {
            for (var k=0; k < i3; k++) {
                out[idx] <== in[i][j][k];
                idx ++;
            }
        }
    }
}

template Flatten4D(i1, i2, i3, i4) {
    signal input in[i1][i2][i3][i4];
    signal output out[i1 * i2 * i3 * i4];

    var idx = 0;
    for (var i=0; i < i1; i++) {
        for (var j=0; j < i2; j++) {
            for (var k=0; k < i3; k++) {
                for (var m=0; m < i4; m++) {
                    out[idx] <== in[i][j][k][m];
                    idx ++;
                }
            }
        }
    }

}
