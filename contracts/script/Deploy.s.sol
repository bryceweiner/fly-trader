// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {VaultScript} from "./VaultScript.sol";
import {FlyVaultStack} from "./FlyVaultStack.sol";

/// @notice Deploys TimelockController + FlyVault implementation + initialized ERC1967Proxy.
/// Env: FLY_TOKEN, OWNER (required); PAUSER (default OWNER); TIMELOCK_DELAY (default 8 days);
///      WITHDRAW_DELAY (default 7 days). On chain 4663 anything but the real $FLY, 8 days and 7 days is refused.
contract Deploy is VaultScript {
    function run() external returns (FlyVaultStack.Deployment memory) {
        return deploy(configFromEnv());
    }

    function configFromEnv() public view returns (FlyVaultStack.Config memory c) {
        c.fly = vm.envAddress("FLY_TOKEN");
        c.owner = vm.envAddress("OWNER");
        c.pauser = vm.envOr("PAUSER", c.owner);
        c.timelockDelay = vm.envOr("TIMELOCK_DELAY", FlyVaultStack.MAINNET_TIMELOCK_DELAY);
        c.withdrawDelay = vm.envOr("WITHDRAW_DELAY", FlyVaultStack.MAINNET_WITHDRAW_DELAY);
    }

    function deploy(FlyVaultStack.Config memory c) public returns (FlyVaultStack.Deployment memory d) {
        FlyVaultStack.enforceChainRules(c);
        vm.startBroadcast();
        (, address deployer,) = vm.readCallers();
        d = FlyVaultStack.deploy(c);
        vm.stopBroadcast();

        FlyVaultStack.check(d, c, deployer);
        require(_implementationOf(address(d.vault)) == address(d.implementation), "check: implementation slot");
        _logDeployment(d, c);
        _writeDeployment(d, c);
    }
}
