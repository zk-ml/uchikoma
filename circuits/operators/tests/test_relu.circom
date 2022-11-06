pragma circom 2.0.3;
include "../ReLU.circom";

template TestReLU() {
  signal input in;
  signal output out;
  component relu = ReLU();
  relu.in <== 2;
  signal mid;
  mid <== relu.out;
  mid === 2;
  component relu1 = ReLU();
  relu1.in <== -2;
  signal mid1;
  mid1 <== relu1.out;
  mid1 === 0;
  out <== mid1;
}

template TestClip() {
  signal input in;
  signal output out;
  component clip = Clip(-1, 3, 1);
  clip.in[0] <== 2;
  signal mid;
  mid <== clip.out[0];
  mid === 2;
  component clip1 = Clip(-1, 3, 1);
  clip1.in[0] <== -2;
  signal mid1;
  mid1 <== clip1.out[0];
  mid1 === -1;
  component clip2 = Clip(-1, 3, 1);
  clip2.in[0] <== 4;
  signal mid2;
  mid2 <== clip2.out[0];
  mid2 === 3;
  out <== mid2;
}

component main = TestClip();
