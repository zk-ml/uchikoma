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

  const [admin] = await ethers.getSigners();
  const AIGC = await ethers.getContractFactory("AIGC");
  const nft = AIGC.connect(admin).attach("0xe2420937E8a0eF61C2543451564bb0695CE9ef8b");

  console.log(await nft.getAllOwned("0xDD63369Cd353f731De50cd2d5F6594Dd7B1083bA"));
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
