pragma circom 2.0.3;

include "../circomlib/compconstant.circom"
include "../circomlib/switcher.circom";

# Operator: look up table
template LUT(iShape, lutShape) {
    signal input in[iShape];
    signal input lut[lutShape];
    signal output out[iShape];

    for (var i=0; i < iShape; i++) {
        component switchers[lutShape+1];

        switchers[0] = Switcher();
        switchers[0].sql <== 1;
        switchers[0].L <== 0;
        switchers[0].R <== 0;

        signal lutBoundChecker[lutShape+1];
        lutBoundChecker[0] = 0;

        for (var j=0; j < lutShape; j++) {
            component isIndex = EqualConstant(j);
            isIndex.in <== in[i];

            switchers[j+1] = Switcher();
            switchers[j+1].sql <== isIndex.out;
            switchers[j+1].L <== switchers[j].outL;
            switchers[j+1].R <== lut[j];

            lutBoundChecker[j+1] <== lutBoundChecker[j] + isIndex.out;
        }

        out[i] <== switchers[lutShape].outL;
        lutBoundChecker[lutShape] === 1;
    }
}
