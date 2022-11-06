require("@nomicfoundation/hardhat-toolbox");
require("@nomiclabs/hardhat-ethers");
// Run abi exporter
require("hardhat-abi-exporter");

// Go to https://www.alchemyapi.io, sign up, create
// a new App in its dashboard, and replace "KEY" with its key
const ALCHEMY_API_KEY = "51T2ZlxpCoe84BuJXGeJdftxZatb_SzV";

const GOERLI_PRIVATE_KEY =
  "702227e8bb0e9339ea5e270c677386a454b71e3056dc60b1d18713a7af53519a";

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    compilers: [
      {
        version: "0.8.9",
        settings: {
          optimizer: {
            enabled: true,
            runs: 1000,
          },
        },
      },
    ],
  },
  abiExporter: [
    {
      path: "./abi/pretty",
      pretty: true,
    },
    {
       path: "./abi/ugly",
       pretty: false,
    },
    {
      path: './abi/json',
      format: "json",
    },
    {
      path: './abi/minimal',
      format: "minimal",
    },
    {
      path: './abi/fullName',
      format: "fullName",
    },
  ],
  networks: {
    hardhat: {
      allowUnlimitedContractSize: true,
      blockGasLimit: 1000000000,
      // accounts:
      //   [
      //     {"privateKey": process.env.PRIVATE_KEY || "", "balance": "10000000000000000000000"},
      //   ],
       chainId: 1337,
    },
    localhost: {
      allowUnlimitedContractSize: true,
      // accounts:
      //   [
      //     {"privateKey": process.env.PRIVATE_KEY || "", "balance": "10000000000000000000000"},
      //   ],
      // chainId: 31337,
    },
    goerli: {
      allowUnlimitedContractSize: true,
      url: `https://eth-goerli.alchemyapi.io/v2/${ALCHEMY_API_KEY}`,
      accounts: [GOERLI_PRIVATE_KEY],
    },
    rinkeby: {
      url: `https://eth-rinkeby.alchemyapi.io/v2/${ALCHEMY_API_KEY}`,
      accounts: ["eb96ea5f0c539df0b4b52783bdae6bb4645234abf100a9e6111a3668769e6cb4"],
    }
  },
};
