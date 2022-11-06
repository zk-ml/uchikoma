pragma circom 2.0.3;
include "../Arithmetic.circom";

template TestAdd(n) {
  signal input a[n];
  signal input b[n];
  signal output out[n];

  component com = Add(n);
  for (var i = 0; i < n; i++) {
    com.a[i] <== a[i];
    com.b[i] <== b[i];
  }
  for (var i = 0; i < n; i++) {
    out[i] <== com.out[i];
  }
}

template TestSub(n) {
  signal input a[n];
  signal input b[n];
  signal output out[n];

  component com = Sub(n);
  for (var i = 0; i < n; i++) {
    com.a[i] <== a[i];
    com.b[i] <== b[i];
  }
  for (var i = 0; i < n; i++) {
    out[i] <== com.out[i];
  }
}

template TestMul(n) {
  signal input a[n];
  signal input b[n];
  signal output out[n];

  component com = Mul(n);
  for (var i = 0; i < n; i++) {
    com.a[i] <== a[i];
    com.b[i] <== b[i];
  }
  for (var i = 0; i < n; i++) {
    out[i] <== com.out[i];
  }
}

component main = TestSub(3);
