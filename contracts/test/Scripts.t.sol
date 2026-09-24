// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {ERC1967Utils} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Utils.sol";
import {Deploy} from "../script/Deploy.s.sol";
import {DeployTestnet} from "../script/DeployTestnet.s.sol";
import {ScheduleUpgrade} from "../script/ScheduleUpgrade.s.sol";
import {ExecuteUpgrade} from "../script/ExecuteUpgrade.s.sol";
import {FlyVaultStack} from "../script/FlyVaultStack.sol";
import {FlyVault} from "../src/FlyVault.sol";
import {MockFLY} from "../src/testnet/MockFLY.sol";

/// @notice Runs the real scripts in-process (nothing is broadcast anywhere). Env vars are process-wide, so everything
///         that sets them lives in the single test below.
contract ScriptsTest is Test {
    function test_scripts_deployScheduleExecute() public {
        address sender = DEFAULT_SENDER; // what vm.startBroadcast() uses under forge test

        // DeployTestnet on anvil's chain id, owned by the broadcaster
        vm.setEnv("OWNER", vm.toString(sender));
        vm.setEnv("TIMELOCK_DELAY", "600");
        vm.setEnv("WITHDRAW_DELAY", "300");
        (MockFLY fly, FlyVaultStack.Deployment memory d) = new DeployTestnet().run();
        assertEq(d.vault.token(), address(fly));
        assertEq(d.vault.WITHDRAW_DELAY(), 300);
        assertEq(d.timelock.getMinDelay(), 600);

        // Deploy.s.sol with the same env, against the mock token
        vm.setEnv("FLY_TOKEN", vm.toString(address(fly)));
        FlyVaultStack.Deployment memory d2 = new Deploy().run();
        assertEq(d2.vault.token(), address(fly));

        // someone locks, then the owner schedules an upgrade through the script
        fly.mint(address(this), 5 ether);
        fly.approve(address(d.vault), 5 ether);
        d.vault.lock(5 ether);

        vm.setEnv("VAULT", vm.toString(address(d.vault)));
        vm.setEnv("TIMELOCK", vm.toString(address(d.timelock)));
        vm.setEnv("SALT", vm.toString(bytes32(uint256(7))));
        (address newImpl, bytes32 opId) = new ScheduleUpgrade().run();
        assertTrue(d.timelock.isOperationPending(opId));
        assertFalse(d.timelock.isOperationReady(opId)); // the rehearsal was rolled back
        assertEq(_impl(address(d.vault)), address(d.implementation));
        assertEq(FlyVault(newImpl).WITHDRAW_DELAY(), 300); // defaulted to the live value

        vm.setEnv("NEW_IMPLEMENTATION", vm.toString(newImpl));
        ExecuteUpgrade exec = new ExecuteUpgrade();
        vm.expectRevert();
        exec.run(); // too early
        skip(600);
        exec.run();
        assertEq(_impl(address(d.vault)), newImpl);
        assertEq(d.vault.locked(address(this)), 5 ether);

        // Deploy.s.sol refuses anything but the frozen parameters on Robinhood Chain mainnet
        vm.chainId(FlyVaultStack.ROBINHOOD_CHAIN_ID);
        Deploy deploy = new Deploy();
        vm.expectRevert(bytes("FlyVaultStack: mainnet FLY_TOKEN must be the real $FLY"));
        deploy.run();
        vm.setEnv("FLY_TOKEN", vm.toString(FlyVaultStack.FLY_MAINNET));
        vm.expectRevert(bytes("FlyVaultStack: mainnet TIMELOCK_DELAY must be 8 days"));
        deploy.run();
        vm.setEnv("TIMELOCK_DELAY", "691200");
        vm.expectRevert(bytes("FlyVaultStack: mainnet WITHDRAW_DELAY must be 7 days"));
        deploy.run();
    }

    function _impl(address proxy) internal view returns (address) {
        return address(uint160(uint256(vm.load(proxy, ERC1967Utils.IMPLEMENTATION_SLOT))));
    }
}
