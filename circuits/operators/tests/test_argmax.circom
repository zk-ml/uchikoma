pragma circom 2.0.3;
include "../ArgMax.circom";

template TestArgMax() {
  signal input in;
  signal output out;
  component agm = ArgMax(3);
  agm.in[0] <== -2;
  agm.in[1] <== -1;
  agm.in[2] <== -3;
  signal mid <== agm.out;
  component agm1 = ArgMax(3);
  agm1.in[0] <== -2;
  agm1.in[1] <== -1;
  agm1.in[2] <== 3;
  signal mid1 <== agm1.out;
  mid1 === 2;
  out <== mid1;
}
component main = TestArgMax();
