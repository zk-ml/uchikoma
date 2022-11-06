pragma circom 2.0.3;

include "../circomlib-matrix/matElemMul.circom";
include "../circomlib-matrix/matElemSum.circom";
include "../util.circom";

template Pad2D (i1, i2, i3, V, padX1, padX2, padY1, padY2) {
    signal input in[i1][i2][i3];
    signal output out[i1][i2+padX1+padX2][i3+padY1+padY2];

    for (var i = 0; i < i1; i++) {
        for (var j = 0; j < i2+padX1+padX2; j++) {
            for (var k = 0; k < i3+padY1+padY2; k++) {
                if ((j < padX1) || ((padX1 + i2) <= j) || (k < padY1) || ((padY1 + i3) <= k)) {
                    out[i][j][k] <== V;
                } else {
                    out[i][j][k] <== in[i][j-padX1][k-padY1];

                }
            }
        }
    }
}

template Conv2D_CHW (C, H, W, F, K, S) {
    signal input in[C][H][W];
    signal input weights[F][C][K][K];
    signal output out[F][(H-K)\S+1][(W-K)\S+1];

    component mul[F][C][(H-K)\S+1][(W-K)\S+1];
    component elemSum[F][C][(H-K)\S+1][(W-K)\S+1];
    component sum[F][(H-K)\S+1][(W-K)\S+1];

    for (var m = 0; m < F; m++) {
        for (var k = 0; k < C; k++) {
            for (var i = 0; i < (H-K)\S+1; i++) {
                for (var j = 0; j < (W-K)\S+1; j++) {
                    mul[m][k][i][j] = matElemMul(K, K);
                    for (var x = 0; x < K; x++) {
                        for (var y = 0; y < K; y++) {
                            mul[m][k][i][j].a[x][y] <== in[k][i*S+x][j*S+y];
                            mul[m][k][i][j].b[x][y] <== weights[m][k][x][y];
                        }
                    }
                    elemSum[m][k][i][j] = matElemSum(K, K);
                    for (var x = 0; x < K; x++) {
                        for (var y = 0; y < K; y++) {
                            elemSum[m][k][i][j].a[x][y] <== mul[m][k][i][j].out[x][y];
                        }
                    }
                }
            }
        }
    }
    for (var m = 0; m < F; m++) {
        for (var i = 0; i < (H-K)\S+1; i++) {
            for (var j = 0; j < (W-K)\S+1; j++) {
                sum[m][i][j] = Sum(C);
                for (var k = 0; k < C; k++) {
                    sum[m][i][j].in[k] <== elemSum[m][k][i][j].out;
                }
                out[m][i][j] <== sum[m][i][j].out;
            }
        }
    }

}

// Conv2D layer with valid padding
template Conv2D (nRows, nCols, nChannels, nFilters, kernelSize, strides) {
    signal input in[nRows][nCols][nChannels];
    signal input weights[kernelSize][kernelSize][nChannels][nFilters];
    signal input bias[nFilters];
    signal output out[(nRows-kernelSize)\strides+1][(nCols-kernelSize)\strides+1][nFilters];

    component mul[(nRows-kernelSize)\strides+1][(nCols-kernelSize)\strides+1][nChannels][nFilters];
    component elemSum[(nRows-kernelSize)\strides+1][(nCols-kernelSize)\strides+1][nChannels][nFilters];
    component sum[(nRows-kernelSize)\strides+1][(nCols-kernelSize)\strides+1][nFilters];

    for (var i=0; i<(nRows-kernelSize)\strides+1; i++) {
        for (var j=0; j<(nCols-kernelSize)\strides+1; j++) {
            for (var k=0; k<nChannels; k++) {
                for (var m=0; m<nFilters; m++) {
                    mul[i][j][k][m] = matElemMul(kernelSize,kernelSize);
                    for (var x=0; x<kernelSize; x++) {
                        for (var y=0; y<kernelSize; y++) {
                            mul[i][j][k][m].a[x][y] <== in[i*strides+x][j*strides+y][k];
                            mul[i][j][k][m].b[x][y] <== weights[x][y][k][m];
                        }
                    }
                    elemSum[i][j][k][m] = matElemSum(kernelSize,kernelSize);
                    for (var x=0; x<kernelSize; x++) {
                        for (var y=0; y<kernelSize; y++) {
                            elemSum[i][j][k][m].a[x][y] <== mul[i][j][k][m].out[x][y];
                        }
                    }
                }
            }
            for (var m=0; m<nFilters; m++) {
                sum[i][j][m] = Sum(nChannels);
                for (var k=0; k<nChannels; k++) {
                    sum[i][j][m].in[k] <== elemSum[i][j][k][m].out;
                }
                out[i][j][m] <== sum[i][j][m].out + bias[m];
            }
        }
    }
}
