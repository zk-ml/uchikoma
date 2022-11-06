pragma circom 2.0.3;

template MulScalar(iShape, sc) {
    signal input in[iShape];
    signal output out[iShape];

    for (var i=0; i < iShape; i++) {
        out[i] <== in[i] * sc;
    }
}

template AddScalar(iShape, sc) {
    signal input in[iShape];
    signal output out[iShape];

    for (var i=0; i < iShape; i++) {
        out[i] <== in[i] + sc;
    }

}

template SubScalar(iShape, sc) {
    signal input in[iShape];
    signal output out[iShape];

    for (var i=0; i < iShape; i++) {
        out[i] <== in[i] - sc;
    }
}
