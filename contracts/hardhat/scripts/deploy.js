// We require the Hardhat Runtime Environment explicitly here. This is optional
// but useful for running the script in a standalone fashion through `node <script>`.
//
// When running the script with `npx hardhat run <script>` you'll find the Hardhat
// Runtime Environment's members available in the global scope.
// const { ethers } = require("ethers");

const fs = require("fs");
const ejs = require("ejs");
// const { unstringifyBigInts } = require("ffjavascript").utils;

function flatten(inp) {
  return [inp[0][0], inp[0][1], inp[1][0], inp[1][1]];
}

async function main() {
  // Hardhat always runs the compile task when running scripts with its command
  // line interface.
  //
  // If this script is run directly using `node` you may want to call compile
  // manually to make sure everything is compiled
  // await hre.run('compile');

  const MintVerifier = await ethers.getContractFactory("MintVerifier");
  const mv = await MintVerifier.deploy();
  await mv.deployed();

  const [admin] = await ethers.getSigners();
  const AIGC = await ethers.getContractFactory("AIGC");
  const maxBatchSize = 1;
  const collectionSize = 42;
  const startPrice = ethers.utils.parseUnits("0", "ether");

  const _nft = await AIGC.deploy(
    maxBatchSize,
    collectionSize,
    // 200,
    // 1,
    startPrice,
    mv.address,
  );
  console.log("Contract address:", _nft.address);
  await _nft.deployed();
  const nft = AIGC.connect(admin).attach(_nft.address);

  const privateData = {
    a: [0,0],
    b: [0,0,0,0],
    c: [0,0]
  };

  
  let publicData = [];
  for (let i = 0; i < 3072; i++) {
    publicData.push(i % 256);
  }

  const mintData = {
    privateData: privateData,
    publicData: publicData,
  }

  await nft.publicMint(mintData);

  console.log(admin.address);
  console.log(await nft.getOwnershipData(0));
  console.log(await nft.tokenURI(0));

  //@ts-ignore
  BigInt.prototype.toJSON = function () {
    return this.toString();
  };

}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
