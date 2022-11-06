pragma circom 2.1.0;

template Upsampling(N, C, H, W, scale) {
    signal input in[N][C][H][W];
    signal output out[N][C][H * scale][W * scale];

    assert(scale > 1);

    for (var i=0; i < N; i++) {
        for (var j=0; j < C; j++) {
            for (var m=0; m < H*scale; m++) {
                for (var n=0; n < W*scale; n++) {
                    out[i][j][m][n] <== in[i][j][m\scale][n\scale];
                }
            }
        }
    }
}
