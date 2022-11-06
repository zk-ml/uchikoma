pragma circom 2.0.3;
include "../util.circom";

template TestAbs() {
  signal input in;
  signal output out;
  component abs = Abs(1);
  abs.in[0] <== 21888242871839275222246405745257275088548364400416034343698204186575808495615;
  out <== abs.out[0];
  out === 2;
}
template TestLessThanFull() {
  signal input in;
  signal output out;
  component ltf = LessThan_Full();
  ltf.a <== 2;
  ltf.b <== -2;
  out <== ltf.out;
  out === 0;
}
component main = TestLessThanFull();
