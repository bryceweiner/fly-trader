// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {console} from "forge-std/Script.sol";
import {TimelockController} from "@openzeppelin/contracts/governance/TimelockController.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts/proxy/utils/UUPSUpgradeable.sol";
import {VaultScript} from "./VaultScript.sol";
import {FlyVault} from "../src/FlyVault.sol";

/// @notice Executes an upgrade scheduled by ScheduleUpgrade.s.sol once its delay has passed. Anyone may execute.
/// Env: NEW_IMPLEMENTATION (required); UPGRADE_CALL, SALT, PREDECESSOR (must match the scheduled values; default
///      empty/0/0); VAULT, TIMELOCK (default: deployments/<chainid>.json).
contract ExecuteUpgrade is VaultScript {
    function run() external {
        address vault = _envOrDeployed("VAULT", "vault");
        TimelockController timelock = TimelockController(payable(_envOrDeployed("TIMELOCK", "timelock")));
        address implementation = vm.envAddress("NEW_IMPLEMENTATION");
        bytes memory upgradeCall = vm.envOr("UPGRADE_CALL", bytes(""));
        bytes32 salt = vm.envOr("SALT", bytes32(0));
        bytes32 predecessor = vm.envOr("PREDECESSOR", bytes32(0));

        bytes memory data = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (implementation, upgradeCall));
        bytes32 id = timelock.hashOperation(vault, 0, data, predecessor, salt);
        console.log("operation id");
        console.logBytes32(id);
        require(timelock.isOperation(id), "ExecuteUpgrade: no such operation (check NEW_IMPLEMENTATION/UPGRADE_CALL/SALT/PREDECESSOR)");
        require(!timelock.isOperationDone(id), "ExecuteUpgrade: already executed");
        require(
            timelock.isOperationReady(id),
            string.concat("ExecuteUpgrade: not ready until ", vm.toString(timelock.getTimestamp(id)))
        );

        vm.startBroadcast();
        timelock.execute(vault, 0, data, predecessor, salt);
        vm.stopBroadcast();

        require(_implementationOf(vault) == implementation, "ExecuteUpgrade: implementation not switched");
        console.log("vault now delegates to", implementation);
        console.log("WITHDRAW_DELAY", FlyVault(vault).WITHDRAW_DELAY());
    }
}
