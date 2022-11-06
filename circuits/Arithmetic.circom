pragma circom 2.0.3;

include "circomlib/comparators.circom";
include "circomlib/switcher.circom";
include "circomlib/bitify.circom";
include "circomlib/sign.circom";

template Mul(iShape) {
    signal input a[iShape];
    signal input b[iShape];
    signal output out[iShape];

    for (var i = 0; i < iShape; i++) {
        // out[i] <== a[i] * b[i] - (1<<64) * (a[i] + b[i] - (1<<64) - 1);
        // out[i] <== a[i] * b[i] + (1<<64)*(1<<64) + (1<<64) - (1<<64) * (a[i] + b[i]);
        out[i] <== a[i] * b[i];
    }
}

template Add(iShape) {
    signal input a[iShape];
    signal input b[iShape];
    signal output out[iShape];

    for (var i = 0; i < iShape; i++) {
        // out[i] <== a[i] + b[i] - (1<<64);
        out[i] <== a[i] + b[i];
    }
}

template Sub(iShape) {
    signal input a[iShape];
    signal input b[iShape];
    signal output out[iShape];

    for (var i = 0; i < iShape; i++) {
        // out[i] <== a[i] + (1<<64) - b[i];
        out[i] <== a[i] - b[i];
    }
}

template Neg(iShape) {
    signal input in[iShape];
    signal output out[iShape];

    for (var i = 0; i < iShape; i++) {
        // out[i] <== 2 * (1<<64) - in[i];
        out[i] <== 0 - in[i];
    }
}
