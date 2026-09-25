// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {UUPSUpgradeable} from "@openzeppelin/contracts/proxy/utils/UUPSUpgradeable.sol";
import {TimelockController} from "@openzeppelin/contracts/governance/TimelockController.sol";
import {FlyVault} from "../src/FlyVault.sol";
import {MockFLY} from "../src/testnet/MockFLY.sol";
import {FlyVaultV2} from "./mocks/FlyVaultV2.sol";
import {VaultHandler} from "./Invariant.t.sol";
import {VaultTest} from "./utils/VaultTest.sol";

/// @dev The invariant handler plus benign upgrades through the timelock at random points.
contract UpgradingHandler is VaultHandler {
    TimelockController internal immutable timelock;
    address internal immutable owner;
    uint256 internal immutable timelockDelay;
    uint256 public upgrades;

    constructor(
        FlyVault v,
        MockFLY f,
        address pauser_,
        uint256 withdrawDelay_,
        TimelockController t,
        address owner_,
        uint256 timelockDelay_
    ) VaultHandler(v, f, pauser_, withdrawDelay_) {
        (timelock, owner, timelockDelay) = (t, owner_, timelockDelay_);
    }

    /// Schedules and executes an upgrade to a fresh V2 (same withdraw delay) through the real timelock.
    function upgrade() external {
        FlyVaultV2 v2 = new FlyVaultV2(withdrawDelay);
        bytes memory data = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (address(v2), ""));
        bytes32 salt = bytes32(upgrades);
        vm.prank(owner);
        timelock.schedule(address(vault), 0, data, bytes32(0), salt, timelockDelay);
        skip(timelockDelay);
        timelock.execute(address(vault), 0, data, bytes32(0), salt);
        upgrades++;
    }
}

/// @notice Stateful fuzzing with upgrades mid-run; after every run, every actor leaves (whatever the pause state)
///         and must get back exactly what they put in, leaving only donations behind.
contract InvariantExitTest is VaultTest {
    UpgradingHandler internal handler;

    function setUp() public override {
        super.setUp();
        handler = new UpgradingHandler(vault, fly, pauser, WITHDRAW_DELAY, timelock, owner, TIMELOCK_DELAY);
        targetContract(address(handler));
    }

    function invariant_solventAcrossUpgrades() public view {
        assertEq(fly.balanceOf(address(vault)), vault.totalLocked() + vault.totalPending() + handler.donated());
        assertEq(vault.token(), address(fly));
    }

    /// Liveness: nobody can be kept in, not by a pause, not by an upgrade, not by other actors.
    function afterInvariant() public {
        uint256 n = handler.actorCount();
        for (uint256 i; i < n; ++i) {
            address a = handler.actors(i);
            uint256 l = vault.locked(a);
            if (l > 0) {
                vm.prank(a);
                vault.requestWithdrawal(l);
            }
        }
        skip(WITHDRAW_DELAY);
        for (uint256 i; i < n; ++i) {
            address a = handler.actors(i);
            uint256[] memory ids = vault.requestsOf(a);
            for (uint256 j; j < ids.length; ++j) {
                (,,, uint8 state) = vault.request(ids[j]);
                if (state == 1) {
                    vm.prank(a);
                    vault.withdraw(ids[j]);
                }
            }
            assertEq(fly.balanceOf(a), handler.lockedBy(a), "actor did not get back exactly what they locked");
            assertEq(vault.locked(a) + vault.pendingOf(a), 0);
        }
        assertEq(vault.totalLocked() + vault.totalPending(), 0);
        assertEq(fly.balanceOf(address(vault)), handler.donated());
    }
}
