### uchikoma

## File structure
- circuits: Circom building blocks for neural network components
- contracts: AIGC NFT protocol with zk-SNARK proof of image creation
- frontend: Simple frontend around model compilation
- integer_only_gan: training an integer-only generative adversarial network & model conversion to TVM
- python: uchikoma python parser from TVM text IR to circom circuits


## How to run

### Step 1: Train an integer-only GAN

Under `integer_only_gan`, run `python dcgan_train.py` to train GAN on ArtBench.

After observing model convergence, run `python dcgan_trace.py` to generate TVM IR stored as `generator_tvm_ir.txt` and `generator_tvm.params`.

### Step 2: Generate Circom circuits

Under `python/zkml`, run `python main.py ./generator_tvm_ir.txt ./generator_tvm.params -o model -in %4 -on %54` to generate Circom circuit used in our demo. You could also inspect other options to generate different decomposition / truncation combinations.

You may then refer to https://github.com/iden3/snarkjs for details on how to generate a zk-SNARK proof and a Solidity Groth16 verifier. This may take a while, even with a beefy machine.

Alternatively, you may interact with our compiler through HTML code under `frontend`.

### Step 3: Run Smart Contracts

Under `contracts/hardhat`, run `npx hardhat compile` to compile the smart contracts used for our improved ERC721 protocol. `contracts/AIGC.sol` is our main NFT contract.

To see how an example interaction with our protocol would look like, run `npx hardhat run scripts/deploy.js`. This will output a base64 string as the tokenURI, which can be easily interpreted by NFT marketplaces. You could also view them using `extract.py`

## Artifacts
