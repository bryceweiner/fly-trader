// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {VaultScript} from "./VaultScript.sol";
import {FlyVaultStack} from "./FlyVaultStack.sol";
import {MockFLY} from "../src/testnet/MockFLY.sol";

/// @notice Rehearsal: MockFLY + the same stack as Deploy.s.sol, with short delays. Only on RH testnet or anvil.
/// Env: OWNER (default: the broadcasting account); PAUSER (default OWNER);
///      TIMELOCK_DELAY (default 600 s); WITHDRAW_DELAY (default 300 s).
contract DeployTestnet is VaultScript {
    function run() external returns (MockFLY fly, FlyVaultStack.Deployment memory d) {
        require(
            block.chainid == FlyVaultStack.ROBINHOOD_TESTNET_CHAIN_ID || block.chainid == FlyVaultStack.ANVIL_CHAIN_ID,
            "DeployTestnet: only chain 46630 or anvil"
        );
        vm.startBroadcast();
        (, address deployer,) = vm.readCallers();
        FlyVaultStack.Config memory c;
        c.owner = vm.envOr("OWNER", deployer);
        c.pauser = vm.envOr("PAUSER", c.owner);
        c.timelockDelay = vm.envOr("TIMELOCK_DELAY", uint256(600));
        c.withdrawDelay = vm.envOr("WITHDRAW_DELAY", uint256(300));
        fly = new MockFLY();
        c.fly = address(fly);
        FlyVaultStack.enforceChainRules(c);
        d = FlyVaultStack.deploy(c);
        vm.stopBroadcast();

        FlyVaultStack.check(d, c, deployer);
        require(_implementationOf(address(d.vault)) == address(d.implementation), "check: implementation slot");
        _logDeployment(d, c);
        _writeDeployment(d, c);
    }
}
