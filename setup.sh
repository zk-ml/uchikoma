
# install circom
git clone https://github.com/iden3/circom.git
cd circom
cargo build --release
cargo install --path circom
cd ..
rm -rf circom

# install snarkjs
npm install -g snarkjs
