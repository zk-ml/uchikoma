pragma circom 2.0.3;
include "../Shift.circom";

template TestRightShift() {
  signal input in;
  signal output out;
  component rs = RightShift(1, 2);
  rs.in[0] <== -16;
  out <== rs.out[0];
  out === -4;
}
component main = TestRightShift();
