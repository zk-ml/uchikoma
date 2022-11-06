pragma circom 2.0.3;

include "../circomlib/comparators.circom";
include "../circomlib/switcher.circom";
include "../util.circom";

template RightShift_Positive(iShape, shiftBit) {
    signal input in[iShape];
    signal output out[iShape];

    signal shiftBitNumber;
    shiftBitNumber <== 1 << shiftBit;

    // reserve some bits, max 252.
    assert(shiftBit <= 248);
    component shiftCheck[iShape];
    for (var i=0; i < iShape; i++) {
        out[i] <-- in[i] >> shiftBit;
        
        shiftCheck[i] = LessThan(252);
        shiftCheck[i].in[0] <== in[i] - (out[i] * shiftBitNumber);
        shiftCheck[i].in[1] <== shiftBitNumber;
        shiftCheck[i].out === 1;
    }
}

// processing negative numbers
template RightShift(iShape, shiftBit) {
    signal input in[iShape];
    signal output out[iShape];
    
    component rightshift_positive[iShape];
    component rightshift_negative[iShape];
    component neg_before[iShape];
    component neg_after[iShape];
    component switcher[iShape];
    component isnegative[iShape];

    for (var i = 0; i < iShape; i++) {
        isnegative[i] = IsNegative();
        isnegative[i].in <== in[i];
        rightshift_positive[i] = RightShift_Positive(1, shiftBit);
        rightshift_positive[i].in[0] <== in[i];
        neg_before[i] = Neg(1);
        neg_before[i].in[0] <== in[i];
        rightshift_negative[i] = RightShift_Positive(1, shiftBit);
        rightshift_negative[i].in[0] <== neg_before[i].out[0];
        neg_after[i] = Neg(1);
        neg_after[i].in[0] <== rightshift_negative[i].out[0];
        switcher[i] = Switcher();
        switcher[i].sel <== isnegative[i].out;
        switcher[i].L <== rightshift_positive[i].out[0];
        switcher[i].R <== neg_after[i].out[0];
        out[i] <== switcher[i].outL;
    } 
    /*
    for (var i = 0; i < iShape; i++) {
        isnegative[i] = IsNegative();
        isnegative[i].in <== in[i];
        rightshift_positive[i] = RightShift(1, shiftBit);
        rightshift_positive[i].in[0] <== in[i] - (1<<64);
        neg_before[i] = Neg(1);
        neg_before[i].in[0] <== in[i];
        rightshift_negative[i] = RightShift(1, shiftBit);
        rightshift_negative[i].in[0] <== neg_before[i].out[0] - (1<<64);
        neg_after[i] = Neg(1);
        neg_after[i].in[0] <== rightshift_negative[i].out[0] + (1<<64);
        switcher[i] = Switcher();
        switcher[i].sel <== isnegative[i].out;
        switcher[i].L <== rightshift_positive[i].out[0] + (1<<64);
        switcher[i].R <== neg_after[i].out[0];
        out[i] <== switcher[i].outL;
    }
    */
    /*
    component rightshift[iShape];
    component neg[iShape];
    component abs[iShape];
    component switcher[iShape];
    component isnegative[iShape];
    for (var i = 0; i < iShape; i++) {
        isnegative[i] = IsNegative();
        isnegative[i].in <== in[i];
        abs[i] = Abs(1);
        abs[i].in[0] <== in[i];
        rightshift[i] = RightShift(1, shiftBit);
        rightshift[i].in[0] <== abs[i].out[0] - (1<<64);
        neg[i] = Neg(1);
        neg[i].in[0] <== rightshift[i].out[0] + (1<<64);
        switcher[i] = Switcher();
        switcher[i].sel <== isnegative[i].out;
        switcher[i].L <== rightshift[i].out[0] + (1<<64);
        switcher[i].R <== neg[i].out[0];
        out[i] <== switcher[i].outL;
    }
    */
}
