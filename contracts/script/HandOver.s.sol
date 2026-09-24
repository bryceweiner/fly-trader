// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console2} from "forge-std/Script.sol";
import {TimelockController} from "@openzeppelin/contracts/governance/TimelockController.sol";
import {IAccessControl} from "@openzeppelin/contracts/access/IAccessControl.sol";
import {FlyVault} from "../src/FlyVault.sol";

/// @notice Moves the vault's owner roles from a temporary key to a new owner (the hardware wallet) through the
///         timelock, so the hand-over is as public and as delayed as an upgrade:
///           timelock PROPOSER + CANCELLER -> NEW_OWNER, then revoked from OLD_OWNER
///           vault PAUSER_ROLE             -> NEW_OWNER, then revoked from OLD_OWNER
///         One batch. MODE=schedule (signed by OLD_OWNER, a proposer) now; MODE=execute (anyone) once the delay passed.
/// Env: VAULT, TIMELOCK, OLD_OWNER, NEW_OWNER, MODE (schedule|execute), SALT (optional, default 0).
contract HandOver is Script {
    function run() external {
        address vault = vm.envAddress("VAULT");
        TimelockController tl = TimelockController(payable(vm.envAddress("TIMELOCK")));
        address oldOwner = vm.envAddress("OLD_OWNER");
        address newOwner = vm.envAddress("NEW_OWNER");
        bytes32 salt = vm.envOr("SALT", bytes32(0));
        string memory mode = vm.envString("MODE");
        (address[] memory targets, uint256[] memory values, bytes[] memory payloads) = batch(vault, address(tl), oldOwner, newOwner);
        bytes32 id = tl.hashOperationBatch(targets, values, payloads, bytes32(0), salt);
        vm.startBroadcast();
        if (keccak256(bytes(mode)) == keccak256("schedule")) {
            tl.scheduleBatch(targets, values, payloads, bytes32(0), salt, tl.getMinDelay());
            console2.log("scheduled hand-over; executable at", block.timestamp + tl.getMinDelay());
        } else if (keccak256(bytes(mode)) == keccak256("execute")) {
            tl.executeBatch(targets, values, payloads, bytes32(0), salt);
            console2.log("hand-over executed");
        } else {
            revert("MODE must be schedule or execute");
        }
        vm.stopBroadcast();
        console2.logBytes32(id);
    }

    function batch(address vault, address tl, address oldOwner, address newOwner)
        public
        view
        returns (address[] memory t, uint256[] memory v, bytes[] memory p)
    {
        require(oldOwner != newOwner && newOwner != address(0), "HandOver: bad owners");
        TimelockController timelock = TimelockController(payable(tl));
        bytes32 proposer = timelock.PROPOSER_ROLE();
        bytes32 canceller = timelock.CANCELLER_ROLE();
        bytes32 pauser = FlyVault(vault).PAUSER_ROLE();
        t = new address[](6);
        v = new uint256[](6);
        p = new bytes[](6);
        // grants first, revocations last: the batch executes atomically either way
        (t[0], p[0]) = (tl, abi.encodeCall(IAccessControl.grantRole, (proposer, newOwner)));
        (t[1], p[1]) = (tl, abi.encodeCall(IAccessControl.grantRole, (canceller, newOwner)));
        (t[2], p[2]) = (vault, abi.encodeCall(IAccessControl.grantRole, (pauser, newOwner)));
        (t[3], p[3]) = (vault, abi.encodeCall(IAccessControl.revokeRole, (pauser, oldOwner)));
        (t[4], p[4]) = (tl, abi.encodeCall(IAccessControl.revokeRole, (canceller, oldOwner)));
        (t[5], p[5]) = (tl, abi.encodeCall(IAccessControl.revokeRole, (proposer, oldOwner)));
    }
}
