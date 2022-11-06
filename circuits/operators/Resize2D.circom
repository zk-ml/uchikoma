pragma circom 2.1.0;

template Resize2D(i1, i2, i3, scale) {
    signal input in[i1][i2][i3];
    signal output out[i1][i2 * scale][i3 * scale];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2 * scale; j++) {
            for (var k = 0; k < i3 * scale; k++) {
                out[i][j][k] <== in[i][j\scale][k\scale];
            }
        }
    }
}
