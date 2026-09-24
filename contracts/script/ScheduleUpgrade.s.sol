// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {console} from "forge-std/Script.sol";
import {TimelockController} from "@openzeppelin/contracts/governance/TimelockController.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts/proxy/utils/UUPSUpgradeable.sol";
import {VaultScript} from "./VaultScript.sol";
import {FlyVault} from "../src/FlyVault.sol";

/// @notice Deploys the current src/FlyVault.sol as a new implementation and schedules
///         `vault.upgradeToAndCall(newImpl, UPGRADE_CALL)` on the timelock with its minimum delay.
///         Must be broadcast by the timelock proposer (the owner's hardware wallet).
/// Env (all optional):
///   VAULT, TIMELOCK      default: deployments/<chainid>.json
///   WITHDRAW_DELAY       constructor arg of the new implementation; default: the live vault's WITHDRAW_DELAY
///   UPGRADE_CALL         calldata run on the proxy right after the upgrade (e.g. a reinitializer); default empty
///   SALT                 bytes32, default 0. Needed only to re-propose an operation identical to one already
///                        executed (same impl + calldata); a cancelled operation can be rescheduled with the same salt.
///   PREDECESSOR          bytes32 operation id that must be executed first; default 0 (none)
/// Before broadcasting anything real, the script rehearses the execution locally (warp past the delay, execute,
/// check the state survived) and reverts if that fails, so a broken upgrade never starts its 8-day clock.
contract ScheduleUpgrade is VaultScript {
    function run() external returns (address implementation, bytes32 operationId) {
        address vault = _envOrDeployed("VAULT", "vault");
        TimelockController timelock = TimelockController(payable(_envOrDeployed("TIMELOCK", "timelock")));
        uint256 withdrawDelay = vm.envOr("WITHDRAW_DELAY", FlyVault(vault).WITHDRAW_DELAY());
        bytes memory upgradeCall = vm.envOr("UPGRADE_CALL", bytes(""));
        bytes32 salt = vm.envOr("SALT", bytes32(0));
        bytes32 predecessor = vm.envOr("PREDECESSOR", bytes32(0));
        uint256 delay = timelock.getMinDelay();

        vm.startBroadcast();
        implementation = address(new FlyVault(withdrawDelay));
        bytes memory data = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (implementation, upgradeCall));
        timelock.schedule(vault, 0, data, predecessor, salt, delay);
        vm.stopBroadcast();

        operationId = timelock.hashOperation(vault, 0, data, predecessor, salt);
        _rehearse(timelock, vault, implementation, data, predecessor, salt);

        console.log("new implementation  ", implementation);
        console.log("operation id");
        console.logBytes32(operationId);
        console.log("eta (unix, from the simulated block)", block.timestamp + delay);
        console.log("execute after the delay with:");
        console.log(string.concat("  NEW_IMPLEMENTATION=", vm.toString(implementation)));
        console.log(string.concat("  UPGRADE_CALL=", vm.toString(upgradeCall)));
        console.log(string.concat("  SALT=", vm.toString(salt), " PREDECESSOR=", vm.toString(predecessor)));
    }

    /// @dev Local-only: executes the operation after warping past the delay, checks it, then rolls the state back.
    function _rehearse(
        TimelockController timelock,
        address vault,
        address implementation,
        bytes memory data,
        bytes32 predecessor,
        bytes32 salt
    ) internal {
        FlyVault v = FlyVault(vault);
        (address token, uint256 locked, uint256 pending) = (v.token(), v.totalLocked(), v.totalPending());
        uint256 snapshot = vm.snapshotState();
        vm.warp(timelock.getTimestamp(timelock.hashOperation(vault, 0, data, predecessor, salt)));
        timelock.execute(vault, 0, data, predecessor, salt);
        require(_implementationOf(vault) == implementation, "rehearsal: implementation not switched");
        require(v.token() == token && v.totalLocked() == locked && v.totalPending() == pending, "rehearsal: state");
        require(v.hasRole(v.UPGRADER_ROLE(), address(timelock)), "rehearsal: timelock lost UPGRADER_ROLE");
        vm.revertToState(snapshot);
        console.log("rehearsal: upgrade executes after the delay and preserves state");
    }
}
