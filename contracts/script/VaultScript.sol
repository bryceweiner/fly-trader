// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console} from "forge-std/Script.sol";
import {VmSafe} from "forge-std/Vm.sol";
import {ERC1967Utils} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Utils.sol";
import {FlyVaultStack} from "./FlyVaultStack.sol";

/// @notice Shared helpers for the deploy and upgrade scripts.
abstract contract VaultScript is Script {
    /// @notice The address a proxy currently delegates to (ERC-1967 implementation slot).
    function _implementationOf(address proxy) internal view returns (address) {
        return address(uint160(uint256(vm.load(proxy, ERC1967Utils.IMPLEMENTATION_SLOT))));
    }

    function _deploymentsPath(bool dryRun) internal view returns (string memory) {
        string memory name = vm.toString(block.chainid);
        return string.concat(vm.projectRoot(), "/deployments/", name, dryRun ? ".dry-run.json" : ".json");
    }

    /// @notice An address from `deployments/<chainid>.json`, used when the matching env var is not set.
    function _envOrDeployed(string memory envName, string memory jsonKey) internal view returns (address a) {
        a = vm.envOr(envName, address(0));
        if (a != address(0)) return a;
        string memory path = _deploymentsPath(false);
        require(vm.isFile(path), string.concat("set ", envName, " or provide ", path));
        // forge-lint: disable-next-line(unsafe-cheatcode)
        a = vm.parseJsonAddress(vm.readFile(path), string.concat(".", jsonKey));
    }

    /// @notice Writes the summary to deployments/<chainid>.json on `--broadcast`, to <chainid>.dry-run.json on a dry
    ///         run, and nowhere under `forge test`. The file is written before forge sends the transactions; the
    ///         receipts in broadcast/<script>/<chainid>/run-latest.json are authoritative.
    function _writeDeployment(FlyVaultStack.Deployment memory d, FlyVaultStack.Config memory c) internal {
        if (vm.isContext(VmSafe.ForgeContext.TestGroup)) return;
        bool dryRun = !vm.isContext(VmSafe.ForgeContext.ScriptBroadcast) && !vm.isContext(VmSafe.ForgeContext.ScriptResume);
        string memory k = "deployment";
        vm.serializeUint(k, "chainId", block.chainid);
        vm.serializeAddress(k, "vault", address(d.vault));
        vm.serializeAddress(k, "implementation", address(d.implementation));
        vm.serializeAddress(k, "timelock", address(d.timelock));
        vm.serializeAddress(k, "token", c.fly);
        vm.serializeAddress(k, "owner", c.owner);
        vm.serializeAddress(k, "pauser", c.pauser);
        vm.serializeUint(k, "timelockDelay", c.timelockDelay);
        vm.serializeUint(k, "withdrawDelay", c.withdrawDelay);
        // Block/time the script was simulated at: a safe lower bound for indexers' start block.
        vm.serializeUint(k, "blockNumber", vm.getBlockNumber());
        string memory json = vm.serializeUint(k, "timestamp", vm.getBlockTimestamp());
        vm.createDir(string.concat(vm.projectRoot(), "/deployments"), true);
        string memory path = _deploymentsPath(dryRun);
        vm.writeJson(json, path);
        console.log("summary written to", path);
    }

    function _logDeployment(FlyVaultStack.Deployment memory d, FlyVaultStack.Config memory c) internal pure {
        console.log("FlyVault (proxy)     ", address(d.vault));
        console.log("FlyVault (impl)      ", address(d.implementation));
        console.log("TimelockController   ", address(d.timelock));
        console.log("token                ", c.fly);
        console.log("owner (proposer)     ", c.owner);
        console.log("pauser               ", c.pauser);
        console.log("timelock delay (s)   ", c.timelockDelay);
        console.log("withdraw delay (s)   ", c.withdrawDelay);
    }
}
