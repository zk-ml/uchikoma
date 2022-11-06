// SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "./ERC721A.sol";
import "@openzeppelin/contracts/utils/Strings.sol";
import "./SSTORE2.sol";

import {MintVerifier} from "./MintVerifier.sol";
import {EBMP} from "./EBMP.sol";
import {Base64} from "./Base64.sol";

struct ZeroKnowledgePrivateData {
    uint256[2] a;
    uint256[4] b;
    uint256[2] c;
}

struct MintData {
    // Zero Knowledge Proof parameter for private data
    ZeroKnowledgePrivateData privateData;
    // Token's public data, as specified per mint circuit
    uint8[3072] publicData;
}

contract AIGC is Ownable, ERC721A, ReentrancyGuard {

    // Specify a start price
    uint256 public currentPrice;

    // zk-SNARK Verifier for mint proof
    address public immutable mintVerifier;

    // Mapping from tokenId to on-chain image
    address[] private content;

    // Event for backend to know that a new NFT is minted
    event Mint(
        address indexed creator,
        uint256 indexed uniqueId,
        uint256 indexed tokenId
    );

    event Revealed(
        uint256 indexed tokenId
    );

    constructor(
        uint256 maxBatchSize_,
        uint256 collectionSize_,
        uint256 startPrice_,
        address mintVerifier_
    ) ERC721A("AIGC", "AIGC", maxBatchSize_, collectionSize_) {
        currentPrice = startPrice_;
        mintVerifier = mintVerifier_;
    }

    modifier callerIsUser() {
        require(tx.origin == msg.sender, "The caller is another contract");
        _;
    }

    function _mint(address creator, MintData calldata data) internal {
        /*
        require(
            MintVerifier(mintVerifier).verifyProof(
                data.privateData.a,
                data.privateData.b,
                data.privateData.c,
                data.publicData
            ),
            "invalid mint proof"
        );
        */

        uint256 tokenId = currentIndex;

        content.push(SSTORE2.write(abi.encodePacked(data.publicData)));

        _safeMint(creator, 1);

        // Emit an event to let the backend know a NFT has been minted
        emit Mint(creator, data.publicData[4], tokenId);
    }

    function publicMint(MintData calldata data)
        external
        payable
        callerIsUser
        nonReentrant
    {
        require(totalSupply() + 1 <= collectionSize, "reached max supply");
        require(msg.value >= currentPrice, "Need to send more ETH.");
        _mint(msg.sender, data);
        refundIfOver(currentPrice);
    }

    function refundIfOver(uint256 price) private {
        require(msg.value >= price, "Need to send more ETH.");
        if (msg.value > price) {
            payable(msg.sender).transfer(msg.value - price);
        }
    }

    string private header =
        '<svg image-rendering="pixelated" preserveAspectRatio="xMinYMin meet" viewBox="0 0 350 350" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" > <image width="100%" height="100%" xlink:href="data:image/bmp;base64,';
    string private footer = '" /> </svg>';

    function _renderOriginal(uint256 tokenId)
        internal
        view
        virtual
        returns (string memory)
    {
        bytes memory rawData = SSTORE2.read(content[tokenId]);
        uint8[] memory image = new uint8[](3072);
        for (uint256 i = 0; i < 3072; i++) {
            image[i] = uint8(rawData[i]);
        }
        string memory enc = EBMP.encodeBMP(image, 32, 32, 3);

        enc = string(abi.encodePacked(header, enc, footer));

        return enc;
    }

    function tokenURI(uint256 tokenId)
        public
        view
        virtual
        override
        returns (string memory)
    {
        require(
            _exists(tokenId),
            "ERC721Metadata: URI query for nonexistent token"
        );
        string memory img;
        img = _renderOriginal(tokenId);

        string memory json = Base64.encode(
            bytes(
                string(
                    abi.encodePacked(
                        '{"name": "', Strings.toString(tokenId) ,'", "image": "data:image/svg+xml;base64,',
                        Base64.encode(bytes(img)), '"}'
                    )
                )
            )
        );
        string memory output = string(
            abi.encodePacked("data:application/json;base64,", json)
        );

        return output;
    }

    function withdrawMoney() external onlyOwner nonReentrant {
        (bool success, ) = msg.sender.call{value: address(this).balance}("");
        require(success, "Transfer failed.");
    }

    function setOwnersExplicit(uint256 quantity)
        external
        onlyOwner
        nonReentrant
    {
        _setOwnersExplicit(quantity);
    }

    function numberMinted(address owner) public view returns (uint256) {
        return _numberMinted(owner);
    }

    function getOwnershipData(uint256 tokenId)
        external
        view
        returns (TokenOwnership memory)
    {
        return ownershipOf(tokenId);
    }

}
