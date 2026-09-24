// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {FlyVault} from "../../src/FlyVault.sol";

/// @notice Upgrade target for tests: same storage (inherited namespace) plus a field in a new namespace.
contract FlyVaultV2 is FlyVault {
    /// @custom:storage-location erc7201:fly.storage.FlyVaultV2
    struct V2Storage {
        string note;
    }

    // keccak256(abi.encode(uint256(keccak256("fly.storage.FlyVaultV2")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant V2_LOCATION = keccak256(abi.encode(uint256(keccak256("fly.storage.FlyVaultV2")) - 1))
        & ~bytes32(uint256(0xff));

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor(uint256 withdrawDelay) FlyVault(withdrawDelay) {}

    function initializeV2(string calldata note_) external reinitializer(2) {
        _v2().note = note_;
    }

    function note() external view returns (string memory) {
        return _v2().note;
    }

    function version() external pure returns (uint256) {
        return 2;
    }

    function _v2() private pure returns (V2Storage storage $) {
        bytes32 slot = V2_LOCATION;
        assembly ("memory-safe") {
            $.slot := slot
        }
    }
}
