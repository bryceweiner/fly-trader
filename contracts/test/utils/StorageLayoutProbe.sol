// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {FlyVault} from "../../src/FlyVault.sol";

/// @notice Never deployed. Solc's storage-layout output ignores ERC-7201 namespaces, so this probe declares the
///         namespaced struct as an ordinary variable to expose its field offsets:
///         forge inspect StorageLayoutProbe storageLayout --json > storage-layout.namespaced.json
contract StorageLayoutProbe {
    FlyVault.FlyVaultStorage internal flyVaultStorage;
}
