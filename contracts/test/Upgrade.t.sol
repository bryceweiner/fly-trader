// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IAccessControl} from "@openzeppelin/contracts/access/IAccessControl.sol";
import {TimelockController} from "@openzeppelin/contracts/governance/TimelockController.sol";
import {Initializable} from "@openzeppelin/contracts/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts/proxy/utils/UUPSUpgradeable.sol";
import {ERC1967Utils} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Utils.sol";
import {IERC1967} from "@openzeppelin/contracts/interfaces/IERC1967.sol";
import {FlyVault} from "../src/FlyVault.sol";
import {FlyVaultV2} from "./mocks/FlyVaultV2.sol";
import {VaultTest} from "./utils/VaultTest.sol";

contract UpgradeTest is VaultTest {
    FlyVaultV2 internal v2;
    bytes32 internal upgraderRole;
    bytes32 internal adminRole;

    function setUp() public override {
        super.setUp();
        v2 = new FlyVaultV2(WITHDRAW_DELAY);
        upgraderRole = vault.UPGRADER_ROLE();
        adminRole = vault.DEFAULT_ADMIN_ROLE();
    }

    function _upgradeCall(address impl, bytes memory data) internal pure returns (bytes memory) {
        return abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (impl, data));
    }

    function _schedule(address target, bytes memory data) internal returns (bytes32 id) {
        vm.prank(owner);
        timelock.schedule(target, 0, data, bytes32(0), bytes32(0), TIMELOCK_DELAY);
        id = timelock.hashOperation(target, 0, data, bytes32(0), bytes32(0));
    }

    function _implementation() internal view returns (address) {
        return address(uint160(uint256(vm.load(address(vault), ERC1967Utils.IMPLEMENTATION_SLOT))));
    }

    // ------------------------------------------------------------------ the only path: timelock after 8 days

    function test_upgrade_viaTimelockAfterDelay_preservesState() public {
        // some state to carry over
        _lock(alice, 100 ether);
        _lock(bob, 50 ether);
        uint256 id1 = _request(alice, 30 ether);
        uint256 id2 = _request(bob, 10 ether);
        vm.prank(bob);
        vault.cancelRequest(id2);
        vm.prank(pauser);
        vault.pause();

        bytes memory data = _upgradeCall(address(v2), abi.encodeCall(FlyVaultV2.initializeV2, ("hello")));
        bytes32 id = _schedule(address(vault), data);

        // not before the delay, by anyone
        skip(TIMELOCK_DELAY - 1);
        vm.expectRevert(
            abi.encodeWithSelector(TimelockController.TimelockUnexpectedOperationState.selector, id, bytes32(uint256(1 << 2)))
        );
        vm.prank(eve);
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));

        skip(1);
        vm.expectEmit(address(vault));
        emit IERC1967.Upgraded(address(v2));
        vm.prank(eve); // executor role is open
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));

        assertEq(_implementation(), address(v2));
        FlyVaultV2 upgraded = FlyVaultV2(address(vault));
        assertEq(upgraded.version(), 2);
        assertEq(upgraded.note(), "hello");
        assertEq(upgraded.token(), address(fly));
        assertEq(upgraded.locked(alice), 70 ether);
        assertEq(upgraded.locked(bob), 50 ether);
        assertEq(upgraded.pendingOf(alice), 30 ether);
        assertEq(upgraded.totalLocked(), 120 ether);
        assertEq(upgraded.totalPending(), 30 ether);
        assertTrue(upgraded.paused());
        assertTrue(upgraded.hasRole(upgraderRole, address(timelock)));
        assertTrue(upgraded.hasRole(upgraded.PAUSER_ROLE(), pauser));
        (address user, uint256 amount, uint64 readyAt, uint8 state) = upgraded.request(id1);
        assertEq(user, alice);
        assertEq(amount, 30 ether);
        assertEq(state, 1);
        assertEq(upgraded.requestsOf(bob)[0], id2);

        // still works after the upgrade; ids continue
        vm.warp(readyAt);
        vm.prank(alice);
        upgraded.withdraw(id1);
        assertEq(fly.balanceOf(alice), 30 ether);
        vm.prank(pauser);
        upgraded.unpause();
        assertEq(_request(bob, 1 ether), 3);

        // reinitializer cannot be replayed
        vm.expectRevert(Initializable.InvalidInitialization.selector);
        upgraded.initializeV2("again");
    }

    function test_upgrade_changesWithdrawDelayForNewRequestsOnly() public {
        _lock(alice, 10 ether);
        uint256 before = _request(alice, 1 ether);
        (,, uint64 readyBefore,) = vault.request(before);

        FlyVaultV2 shorter = new FlyVaultV2(3 days);
        bytes memory data = _upgradeCall(address(shorter), "");
        _schedule(address(vault), data);
        skip(TIMELOCK_DELAY);
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));

        assertEq(vault.WITHDRAW_DELAY(), 3 days);
        (,, uint64 readyStill,) = vault.request(before);
        assertEq(readyStill, readyBefore);
        uint256 afterId = _request(alice, 1 ether);
        (,, uint64 readyAfter,) = vault.request(afterId);
        assertEq(readyAfter, block.timestamp + 3 days);
    }

    function test_upgrade_directCallsRevert() public {
        address[5] memory callers = [owner, pauser, eve, alice, address(this)];
        for (uint256 i; i < callers.length; ++i) {
            vm.prank(callers[i]);
            vm.expectRevert(
                abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, callers[i], upgraderRole)
            );
            vault.upgradeToAndCall(address(v2), "");
        }
        assertEq(_implementation(), address(implementation));
    }

    function test_grantRole_onlyViaTimelock() public {
        address[5] memory callers = [owner, pauser, eve, alice, address(this)];
        bytes32[3] memory roles = [upgraderRole, adminRole, vault.PAUSER_ROLE()];
        for (uint256 i; i < callers.length; ++i) {
            for (uint256 j; j < roles.length; ++j) {
                vm.prank(callers[i]);
                vm.expectRevert(
                    abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, callers[i], adminRole)
                );
                vault.grantRole(roles[j], callers[i]);
            }
        }
        // the timelock can, after the delay (e.g. rotate the pauser)
        bytes memory data = abi.encodeCall(IAccessControl.grantRole, (vault.PAUSER_ROLE(), bob));
        _schedule(address(vault), data);
        skip(TIMELOCK_DELAY);
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));
        assertTrue(vault.hasRole(vault.PAUSER_ROLE(), bob));
    }

    function test_timelock_onlyOwnerSchedules_minDelayEnforced() public {
        bytes memory data = _upgradeCall(address(v2), "");
        bytes32 proposer = timelock.PROPOSER_ROLE();
        address[3] memory others = [pauser, eve, address(this)];
        for (uint256 i; i < others.length; ++i) {
            vm.prank(others[i]);
            vm.expectRevert(
                abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, others[i], proposer)
            );
            timelock.schedule(address(vault), 0, data, bytes32(0), bytes32(0), TIMELOCK_DELAY);
        }
        vm.prank(owner);
        vm.expectRevert(
            abi.encodeWithSelector(TimelockController.TimelockInsufficientDelay.selector, TIMELOCK_DELAY - 1, TIMELOCK_DELAY)
        );
        timelock.schedule(address(vault), 0, data, bytes32(0), bytes32(0), TIMELOCK_DELAY - 1);
    }

    function test_timelock_ownerCancels() public {
        bytes memory data = _upgradeCall(address(v2), "");
        bytes32 id = _schedule(address(vault), data);
        bytes32 canceller = timelock.CANCELLER_ROLE();
        vm.prank(eve);
        vm.expectRevert(abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, eve, canceller));
        timelock.cancel(id);
        vm.prank(owner);
        timelock.cancel(id);
        skip(TIMELOCK_DELAY);
        vm.expectRevert(
            abi.encodeWithSelector(TimelockController.TimelockUnexpectedOperationState.selector, id, bytes32(uint256(1 << 2)))
        );
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));
    }

    function test_timelock_updateDelayMustWaitTheDelay() public {
        vm.prank(owner);
        vm.expectRevert(abi.encodeWithSelector(TimelockController.TimelockUnauthorizedCaller.selector, owner));
        timelock.updateDelay(1);

        bytes memory data = abi.encodeCall(TimelockController.updateDelay, (1 hours));
        bytes32 id = _schedule(address(timelock), data);
        skip(TIMELOCK_DELAY - 1);
        vm.expectRevert(
            abi.encodeWithSelector(TimelockController.TimelockUnexpectedOperationState.selector, id, bytes32(uint256(1 << 2)))
        );
        timelock.execute(address(timelock), 0, data, bytes32(0), bytes32(0));
        skip(1);
        timelock.execute(address(timelock), 0, data, bytes32(0), bytes32(0));
        assertEq(timelock.getMinDelay(), 1 hours);
    }

    function test_timelock_hasNoExternalAdmin() public {
        bytes32 tAdmin = timelock.DEFAULT_ADMIN_ROLE();
        assertTrue(timelock.hasRole(tAdmin, address(timelock)));
        assertFalse(timelock.hasRole(tAdmin, owner));
        assertFalse(timelock.hasRole(tAdmin, address(this))); // deployer
        bytes32 proposer = timelock.PROPOSER_ROLE();
        vm.prank(owner);
        vm.expectRevert(abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, owner, tAdmin));
        timelock.grantRole(proposer, eve);
    }

    // ------------------------------------------------------------------ initializers and UUPS guards

    function test_initializer_lockedOnImplementation() public {
        vm.expectRevert(Initializable.InvalidInitialization.selector);
        implementation.initialize(address(fly), eve, eve);
    }

    function test_initializer_cannotRerunOnProxy() public {
        vm.prank(eve);
        vm.expectRevert(Initializable.InvalidInitialization.selector);
        vault.initialize(address(fly), eve, eve);
    }

    function test_upgradeOnImplementationDirectlyReverts() public {
        vm.prank(address(timelock));
        vm.expectRevert(UUPSUpgradeable.UUPSUnauthorizedCallContext.selector);
        implementation.upgradeToAndCall(address(v2), "");
    }

    function test_upgradeToNonUupsReverts() public {
        bytes memory data = _upgradeCall(address(fly), "");
        _schedule(address(vault), data);
        skip(TIMELOCK_DELAY);
        vm.expectRevert(abi.encodeWithSelector(ERC1967Utils.ERC1967InvalidImplementation.selector, address(fly)));
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));
    }
}
