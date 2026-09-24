// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {TimelockController} from "@openzeppelin/contracts/governance/TimelockController.sol";
import {FlyVault} from "../../src/FlyVault.sol";
import {MockFLY} from "../../src/testnet/MockFLY.sol";
import {FlyVaultStack} from "../../script/FlyVaultStack.sol";

/// @notice Deploys the production stack (same code path as Deploy.s.sol) with mainnet delays and MockFLY.
abstract contract VaultTest is Test {
    uint256 internal constant TIMELOCK_DELAY = 8 days;
    uint256 internal constant WITHDRAW_DELAY = 7 days;

    address internal owner = makeAddr("owner");
    address internal pauser = makeAddr("pauser");
    address internal alice = makeAddr("alice");
    address internal bob = makeAddr("bob");
    address internal eve = makeAddr("eve");

    MockFLY internal fly;
    FlyVault internal vault;
    FlyVault internal implementation;
    TimelockController internal timelock;
    FlyVaultStack.Config internal cfg;

    function setUp() public virtual {
        vm.warp(1_790_000_000);
        fly = new MockFLY();
        cfg = FlyVaultStack.Config({
            fly: address(fly),
            owner: owner,
            pauser: pauser,
            timelockDelay: TIMELOCK_DELAY,
            withdrawDelay: WITHDRAW_DELAY
        });
        FlyVaultStack.Deployment memory d = FlyVaultStack.deploy(cfg);
        FlyVaultStack.check(d, cfg, address(this));
        (timelock, implementation, vault) = (d.timelock, d.implementation, d.vault);
    }

    function _fund(address user, uint256 amount) internal {
        fly.mint(user, amount);
        vm.prank(user);
        fly.approve(address(vault), type(uint256).max);
    }

    function _lock(address user, uint256 amount) internal {
        _fund(user, amount);
        vm.prank(user);
        vault.lock(amount);
    }

    function _request(address user, uint256 amount) internal returns (uint256 id) {
        vm.prank(user);
        id = vault.requestWithdrawal(amount);
    }
}
