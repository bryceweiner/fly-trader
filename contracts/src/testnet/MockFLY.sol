// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/// @title MockFLY
/// @notice Testnet stand-in for $FLY (18 decimals) with a public faucet, for the vault rehearsal only.
contract MockFLY is ERC20 {
    /// @notice Largest amount one `mint` call may create (1M mFLY); guards against fat-fingered or overflowing mints.
    uint256 public constant MAX_MINT = 1_000_000 ether;

    error MintCapExceeded(uint256 amount, uint256 cap);

    constructor() ERC20("Mock FLY", "mFLY") {}

    /// @notice Public faucet: anyone may mint up to `MAX_MINT` per call.
    function mint(address to, uint256 amount) external {
        if (amount > MAX_MINT) revert MintCapExceeded(amount, MAX_MINT);
        _mint(to, amount);
    }
}
