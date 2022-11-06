#!/bin/bash
scheme=groth16
scheme=plonk
pot=pot16_final.ptau
pot=pot20.ptau
pot=pot12_final.ptau
pot=pot28.ptau

cd `dirname $0`
[ $# -lt 1 ] && exit 0
circuit=$1

circom ${circuit}.circom --r1cs --wasm --sym --c
cd ${circuit}_js
node generate_witness.js ${circuit}.wasm ../${circuit}_input.json ../witness.wtns
cd ..
snarkjs $scheme setup ${circuit}.r1cs $pot circuit_final.zkey
snarkjs zkey export verificationkey circuit_final.zkey verification_key.json
snarkjs $scheme prove circuit_final.zkey witness.wtns proof.json public.json
snarkjs $scheme verify verification_key.json public.json proof.json
snarkjs zkey export solidityverifier circuit_final.zkey verifier.sol
snarkjs generatecall
