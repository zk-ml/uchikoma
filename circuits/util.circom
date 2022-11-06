pragma circom 2.0.3;

include "./circomlib/sign.circom";
include "./circomlib/bitify.circom";
include "./circomlib/comparators.circom";
include "Arithmetic.circom";

template Uint82Num(pack) {
    signal input in[pack];
    signal output out;
    var lc1=0;

    for (var i = 0; i<pack; i++) {
        lc1 += in[i] * 256**i;
    }

    lc1 ==> out;
}

template Num2Uint8(pack) {
    signal input in;
    signal output out[pack];
    var lc1 = 0;

    component rps[pack];

    var e2 = 1;
    for (var i = 0; i<pack; i++) {

        out[i] <-- (in >> i * 8) & 255;

        rps[i] = LessThan(9);
        rps[i].in[0] <== out[i];
        rps[i].in[1] <== 256;
        rps[i].out === 1;

        lc1 += out[i] * e2;
        e2 = e2 * 256;
    }

    lc1 === in;
}

template IsNegative() {
    signal input in;
    signal output out;

    component num2Bits = Num2Bits(254);
    num2Bits.in <== in;
    component sign = Sign();
    
    for (var i = 0; i < 254; i++) {
        sign.in[i] <== num2Bits.out[i];
    }

    out <== sign.sign;
}
/*
template IsNegative() {
    signal input in;
    signal output out;

    component num2Bits = Num2Bits(254);
    num2Bits.in <== in;

    out <== 1 - num2Bits.out[64];
}
*/

template IsPositive() {
    signal input in;
    signal output out;

    component neg = IsNegative();
    neg.in <== in;
    out <== 1 - neg.out;
}

// less than, considering negative numbers
template LessThan_Full() {
    signal input a;
    signal input b;
    signal output out;

    component sub = Sub(1);
    component isneg = IsNegative();
    sub.a[0] <== a;
    sub.b[0] <== b;
    isneg.in <== sub.out[0];
    out <== isneg.out;
}

template LessEqThan_Full() {
    signal input a;
    signal input b;
    signal output out;

    component lt = LessThan_Full();

    lt.a <== a;
    lt.b <== b+1;
    lt.out ==> out;
}

template GreaterThan_Full() {
    signal input a;
    signal input b;
    signal output out;

    component lt = LessThan_Full();

    lt.a <== b;
    lt.b <== a;
    lt.out ==> out;
}

template Abs(iShape) {
    signal input in[iShape];
    signal output out[iShape];
    /*
    component com[iShape];
    component switcher[iShape];
    for (var i = 0; i < iShape; i++) {
        com[i] = LessThan(65);
        com[i].in[0] <== in[i];
        com[i].in[1] <== (1<<64);
        switcher[i] = Switcher();
        switcher[i].sel <== com[i].out;
        switcher[i].L <== in[i];
        switcher[i].R <== 2 * (1<<64) - in[i];
        out[i] <== switcher[i].outL;
    }*/
    component com[iShape];
    component switcher[iShape];
    component neg[iShape];
    for (var i = 0; i < iShape; i++) {
        com[i] = IsNegative();
        com[i].in <== in[i];
        neg[i] = Neg(1);
        neg[i].in[0] <== in[i];
        switcher[i] = Switcher();
        switcher[i].sel <== com[i].out;
        switcher[i].L <== in[i];
        switcher[i].R <== neg[i].out[0];
        out[i] <== switcher[i].outL;
    }
}

template Sum(nInputs) {
    signal input in[nInputs];
    signal output out;

    signal partialSum[nInputs];
    partialSum[0] <== in[0];
    component add[nInputs];

    for (var i=1; i<nInputs; i++) {
        add[i] = Add(1);
        add[i].a[0] <== partialSum[i-1];
        add[i].b[0] <== in[i];
        // partialSum[i] <== partialSum[i-1] + in[i];
        partialSum[i] <== add[i].out[0];
    }

    out <== partialSum[nInputs-1];
}
