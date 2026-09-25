// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IAccessControl} from "@openzeppelin/contracts/access/IAccessControl.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {ERC1967Utils} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Utils.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts/proxy/utils/UUPSUpgradeable.sol";
import {TimelockController} from "@openzeppelin/contracts/governance/TimelockController.sol";
import {PausableUpgradeable} from "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";
import {FlyVault} from "../src/FlyVault.sol";
import {FlyVaultV2} from "./mocks/FlyVaultV2.sol";
import {VaultTest} from "./utils/VaultTest.sol";
import {EvilVault} from "./RedTeam.t.sol";

/// @notice Second round of adversarial cases (2026-09-25 audit): value transfers, allowance hijacking, request spam,
///         pauser loss, batched timelock tricks, cross-vault replay, and the late-locker side of the upgrade window.
contract RedTeamExtraTest is VaultTest {
    // ------------------------------------------------------------------ value and unknown calls

    /// No function is payable and there is no receive/fallback behind the proxy: ETH can never get stuck in the vault.
    function test_redteam_ethIsRefusedEverywhere() public {
        vm.deal(eve, 10 ether);
        vm.startPrank(eve);
        (bool ok,) = address(vault).call{value: 1 ether}("");
        assertFalse(ok, "plain ETH transfer accepted");
        (ok,) = address(vault).call{value: 1 ether}(hex"deadbeef");
        assertFalse(ok, "unknown selector with value accepted");
        (ok,) = address(vault).call{value: 1}(abi.encodeCall(FlyVault.pause, ()));
        assertFalse(ok, "non-payable function accepted value");
        (ok,) = address(vault).call(hex"deadbeef");
        assertFalse(ok, "unknown selector accepted");
        vm.stopPrank();
        assertEq(address(vault).balance, 0);
    }

    // ------------------------------------------------------------------ allowances and other people's requests

    /// Alice's unlimited approval to the vault gives nobody else a way to move her tokens: every entry point acts on
    /// msg.sender only, and her request ids are useless to anyone else.
    function testFuzz_redteam_approvalAndIdsCannotBeHijacked(uint96 aliceAmt, uint96 eveAmt, uint256 guess) public {
        aliceAmt = uint96(bound(aliceAmt, 2, fly.MAX_MINT()));
        eveAmt = uint96(bound(eveAmt, 1, fly.MAX_MINT()));
        _fund(alice, aliceAmt); // unlimited approval
        vm.prank(alice);
        vault.lock(aliceAmt / 2);
        uint256 aliceId = _request(alice, aliceAmt / 4 == 0 ? 1 : aliceAmt / 4);
        uint256 aliceWallet = fly.balanceOf(alice);
        uint256 aliceLocked = vault.locked(alice);
        uint256 alicePending = vault.pendingOf(alice);

        _lock(eve, eveAmt);
        skip(WITHDRAW_DELAY);
        vm.startPrank(eve);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.NotRequestOwner.selector, aliceId));
        vault.withdraw(aliceId);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.NotRequestOwner.selector, aliceId));
        vault.cancelRequest(aliceId);
        // a guessed id is either unknown or someone else's
        uint256 g = bound(guess, 0, 1000);
        if (g != 0 && g != aliceId) {
            vm.expectRevert(abi.encodeWithSelector(FlyVault.UnknownRequest.selector, g));
            vault.withdraw(g);
        }
        // eve can only ever take out her own locked amount
        uint256 eveId = vault.requestWithdrawal(eveAmt);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.InsufficientLocked.selector, 1, 0));
        vault.requestWithdrawal(1);
        skip(WITHDRAW_DELAY);
        vault.withdraw(eveId);
        vm.stopPrank();

        assertEq(fly.balanceOf(eve), eveAmt);
        assertEq(fly.balanceOf(alice), aliceWallet);
        assertEq(vault.locked(alice), aliceLocked);
        assertEq(vault.pendingOf(alice), alicePending);
    }

    /// Thousands of requests from one address cost only the spammer: the vault never iterates over requests, so
    /// another holder's request and withdraw cost exactly the same (cold storage) gas with or without the spam.
    function test_redteam_requestSpamOnlyCostsTheSpammer() public {
        _lock(alice, 10 ether);
        _lock(eve, 1 ether);
        _lock(bob, 1 ether);
        _request(bob, 1); // counters are non-zero in both runs, so only the spam differs
        uint256 snap = vm.snapshotState();
        (uint256 reqBefore, uint256 wBefore) = _aliceRequestAndWithdrawGas();
        vm.revertToState(snap);

        vm.startPrank(eve);
        for (uint256 i; i < 2000; ++i) {
            vault.requestWithdrawal(1);
        }
        vm.stopPrank();
        assertEq(vault.requestsOf(eve).length, 2000);
        (uint256 reqAfter, uint256 wAfter) = _aliceRequestAndWithdrawGas();
        assertEq(reqAfter, reqBefore);
        assertEq(wAfter, wBefore);
        assertEq(vault.requestsOf(alice).length, 1);
    }

    function _aliceRequestAndWithdrawGas() internal returns (uint256 req, uint256 w) {
        vm.cool(address(vault));
        vm.cool(address(fly));
        vm.prank(alice);
        uint256 g0 = gasleft();
        uint256 id = vault.requestWithdrawal(1 ether);
        req = g0 - gasleft();
        skip(WITHDRAW_DELAY);
        vm.cool(address(vault));
        vm.cool(address(fly));
        vm.prank(alice);
        g0 = gasleft();
        vault.withdraw(id);
        w = g0 - gasleft();
    }

    // ------------------------------------------------------------------ pauser

    /// If the pauser pauses and then loses or renounces its key, holders can still leave; only lock and cancel stay
    /// blocked until the timelock appoints a new pauser (an ordinary 8-day operation).
    function test_redteam_pauserLossCannotTrapFunds() public {
        _lock(alice, 5 ether);
        uint256 pending = _request(alice, 2 ether);
        vm.startPrank(pauser);
        vault.pause();
        vault.renounceRole(vault.PAUSER_ROLE(), pauser);
        vm.expectRevert();
        vault.unpause();
        vm.stopPrank();

        // leaving still works: the old pending request and a brand-new one
        uint256 fresh = _request(alice, 3 ether);
        skip(WITHDRAW_DELAY);
        vm.startPrank(alice);
        vault.withdraw(pending);
        vault.withdraw(fresh);
        vm.stopPrank();
        assertEq(fly.balanceOf(alice), 5 ether);
        assertEq(fly.balanceOf(address(vault)), 0);

        // recovery path: the timelock grants a new pauser
        address newPauser = makeAddr("newPauser");
        bytes memory grant = abi.encodeCall(IAccessControl.grantRole, (vault.PAUSER_ROLE(), newPauser));
        vm.prank(owner);
        timelock.schedule(address(vault), 0, grant, bytes32(0), bytes32(0), TIMELOCK_DELAY);
        skip(TIMELOCK_DELAY);
        timelock.execute(address(vault), 0, grant, bytes32(0), bytes32(0));
        vm.prank(newPauser);
        vault.unpause();
        _lock(alice, 1 ether);
        assertEq(vault.locked(alice), 1 ether);
    }

    function test_redteam_pauseStateMachine() public {
        bytes32 pauserRole = vault.PAUSER_ROLE();
        vm.expectRevert(
            abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, eve, pauserRole)
        );
        vm.prank(eve);
        vault.pause();
        vm.prank(pauser);
        vm.expectRevert(PausableUpgradeable.ExpectedPause.selector);
        vault.unpause();
        vm.startPrank(pauser);
        vault.pause();
        vm.expectRevert(PausableUpgradeable.EnforcedPause.selector);
        vault.pause();
        vm.stopPrank();
        vm.expectRevert(
            abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, eve, pauserRole)
        );
        vm.prank(eve);
        vault.unpause();
        // the timelock is DEFAULT_ADMIN, not a pauser: it cannot pause/unpause directly either
        vm.prank(address(timelock));
        vm.expectRevert(
            abi.encodeWithSelector(
                IAccessControl.AccessControlUnauthorizedAccount.selector, address(timelock), pauserRole
            )
        );
        vault.unpause();
    }

    /// Nobody but the admin (the timelock) can grant or revoke roles; the pauser and holders cannot promote themselves.
    function test_redteam_rolesCannotBeSelfGranted() public {
        bytes32 admin = vault.DEFAULT_ADMIN_ROLE();
        bytes32 pauserRole = vault.PAUSER_ROLE();
        bytes32[3] memory roles = [admin, vault.UPGRADER_ROLE(), pauserRole];
        address[3] memory who = [eve, pauser, owner];
        for (uint256 i; i < who.length; ++i) {
            for (uint256 j; j < roles.length; ++j) {
                vm.prank(who[i]);
                vm.expectRevert(
                    abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, who[i], admin)
                );
                vault.grantRole(roles[j], who[i]);
            }
            vm.prank(who[i]);
            vm.expectRevert(
                abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, who[i], admin)
            );
            vault.revokeRole(pauserRole, pauser);
        }
        // renouncing on someone else's behalf is refused
        vm.prank(eve);
        vm.expectRevert(IAccessControl.AccessControlBadConfirmation.selector);
        vault.renounceRole(pauserRole, pauser);
        assertEq(vault.getRoleAdmin(pauserRole), admin);
        assertEq(vault.getRoleAdmin(vault.UPGRADER_ROLE()), admin);
    }

    // ------------------------------------------------------------------ timelock tricks

    /// A batch that first zeroes the timelock delay and then upgrades is still one 8-day operation: batching cannot
    /// make the upgrade land sooner, and the batch cannot be split to run the upgrade half alone.
    function test_redteam_batchedDelayZeroPlusUpgradeStillWaits() public {
        EvilVault evil = new EvilVault();
        address[] memory targets = new address[](2);
        uint256[] memory values = new uint256[](2);
        bytes[] memory payloads = new bytes[](2);
        targets[0] = address(timelock);
        payloads[0] = abi.encodeCall(TimelockController.updateDelay, (0));
        targets[1] = address(vault);
        payloads[1] = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (address(evil), ""));

        vm.prank(owner);
        vm.expectRevert(
            abi.encodeWithSelector(TimelockController.TimelockInsufficientDelay.selector, 0, TIMELOCK_DELAY)
        );
        timelock.scheduleBatch(targets, values, payloads, bytes32(0), bytes32(0), 0);

        vm.prank(owner);
        timelock.scheduleBatch(targets, values, payloads, bytes32(0), bytes32(0), TIMELOCK_DELAY);
        skip(TIMELOCK_DELAY - 1);
        vm.expectRevert();
        timelock.executeBatch(targets, values, payloads, bytes32(0), bytes32(0));
        vm.expectRevert(); // the upgrade half on its own was never scheduled
        timelock.execute(address(vault), 0, payloads[1], bytes32(0), bytes32(0));
        skip(1);
        timelock.executeBatch(targets, values, payloads, bytes32(0), bytes32(0));
        assertEq(_implementation(), address(evil));
    }

    /// A predecessor that never executes blocks the dependent op; it cannot be used to jump the queue.
    function test_redteam_predecessorCannotShortcut() public {
        FlyVaultV2 v2 = new FlyVaultV2(WITHDRAW_DELAY);
        bytes memory up = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (address(v2), ""));
        bytes32 bogus = keccak256("never scheduled");
        vm.prank(owner);
        timelock.schedule(address(vault), 0, up, bogus, bytes32(0), TIMELOCK_DELAY);
        skip(TIMELOCK_DELAY);
        vm.expectRevert(abi.encodeWithSelector(TimelockController.TimelockUnexecutedPredecessor.selector, bogus));
        timelock.execute(address(vault), 0, up, bogus, bytes32(0));
    }

    /// Trust boundary, documented: the exit window protects only holders who request within
    /// TIMELOCK_DELAY - WITHDRAW_DELAY (1 day) of the schedule. Someone who locks after that is trapped if the op is
    /// malicious. The site must therefore warn before `lock` while any vault/timelock operation is pending.
    function test_redteam_lateLockerAfterAScheduledUpgradeCannotExitInTime() public {
        EvilVault evil = new EvilVault();
        bytes memory data = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (address(evil), ""));
        vm.prank(owner);
        timelock.schedule(address(vault), 0, data, bytes32(0), bytes32(0), TIMELOCK_DELAY);

        skip(TIMELOCK_DELAY - WITHDRAW_DELAY + 1); // just past the last safe moment
        _lock(alice, 7 ether);
        uint256 id = _request(alice, 7 ether);
        skip(WITHDRAW_DELAY - 1); // the op is ready one second before alice's request is
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));
        EvilVault(address(vault)).sweep(eve);
        assertEq(fly.balanceOf(eve), 7 ether);
        skip(1);
        vm.prank(alice);
        vm.expectRevert(); // nothing left to pay her with
        vault.withdraw(id);
    }

    /// The owner (proposer/canceller) cannot execute or re-time a cancelled operation.
    function test_redteam_cancelledOperationIsDead() public {
        EvilVault evil = new EvilVault();
        bytes memory data = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (address(evil), ""));
        vm.startPrank(owner);
        timelock.schedule(address(vault), 0, data, bytes32(0), bytes32(0), TIMELOCK_DELAY);
        bytes32 opId = timelock.hashOperation(address(vault), 0, data, bytes32(0), bytes32(0));
        timelock.cancel(opId);
        vm.stopPrank();
        skip(TIMELOCK_DELAY);
        vm.expectRevert();
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));
        vm.prank(eve);
        vm.expectRevert(); // eve is not a canceller
        timelock.cancel(opId);
    }

    // ------------------------------------------------------------------ cross-instance and upgrade edge cases

    /// Two vaults on one implementation keep fully separate books: an id from one means nothing in the other.
    function test_redteam_crossVaultIdsDoNotCarryOver() public {
        FlyVault other = FlyVault(
            address(
                new ERC1967Proxy(
                    address(implementation),
                    abi.encodeCall(FlyVault.initialize, (address(fly), pauser, address(timelock)))
                )
            )
        );
        _lock(alice, 3 ether);
        uint256 id = _request(alice, 3 ether);
        skip(WITHDRAW_DELAY);
        vm.prank(alice);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.UnknownRequest.selector, id));
        other.withdraw(id);
        assertEq(fly.balanceOf(address(other)), 0);
    }

    /// Upgrading keeps every pending request withdrawable at its original readyAt, and V2 cannot be re-initialized
    /// by anyone after the timelock's own call.
    function test_redteam_upgradeKeepsPendingRequestsAndBlocksReinit() public {
        _lock(alice, 4 ether);
        uint256 id = _request(alice, 4 ether);
        (,, uint64 readyAt,) = vault.request(id);
        FlyVaultV2 v2 = new FlyVaultV2(1 hours);
        bytes memory data = abi.encodeCall(
            UUPSUpgradeable.upgradeToAndCall, (address(v2), abi.encodeCall(FlyVaultV2.initializeV2, ("v2")))
        );
        vm.prank(owner);
        timelock.schedule(address(vault), 0, data, bytes32(0), bytes32(0), TIMELOCK_DELAY);
        skip(TIMELOCK_DELAY);
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));
        (,, uint64 readyAfter, uint8 state) = vault.request(id);
        assertEq(readyAfter, readyAt);
        assertEq(state, 1);
        vm.prank(eve);
        vm.expectRevert();
        FlyVaultV2(address(vault)).initializeV2("pwned");
        assertEq(FlyVaultV2(address(vault)).note(), "v2");
        vm.prank(alice);
        vault.withdraw(id);
        assertEq(fly.balanceOf(alice), 4 ether);
    }

    /// Extreme amounts: locking a huge balance works exactly and the totals never wrap.
    function test_redteam_extremeAmountsAccountExactly() public {
        HugeToken h = new HugeToken();
        FlyVault v = FlyVault(
            address(
                new ERC1967Proxy(
                    address(implementation),
                    abi.encodeCall(FlyVault.initialize, (address(h), pauser, address(timelock)))
                )
            )
        );
        uint256 half = type(uint256).max / 2;
        h.mint(alice, half);
        h.mint(bob, half);
        vm.prank(alice);
        h.approve(address(v), type(uint256).max);
        vm.prank(bob);
        h.approve(address(v), type(uint256).max);
        vm.prank(alice);
        v.lock(half);
        vm.prank(bob);
        v.lock(half);
        assertEq(v.totalLocked(), half * 2);
        vm.prank(alice);
        uint256 id = v.requestWithdrawal(half);
        assertEq(v.totalLocked() + v.totalPending(), half * 2);
        skip(WITHDRAW_DELAY);
        vm.prank(alice);
        v.withdraw(id);
        assertEq(h.balanceOf(alice), half);
    }

    function _implementation() internal view returns (address) {
        return address(uint160(uint256(vm.load(address(vault), ERC1967Utils.IMPLEMENTATION_SLOT))));
    }
}

/// @dev Plain ERC-20 with an uncapped mint, for extreme-amount tests.
contract HugeToken {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        if (allowance[from][msg.sender] != type(uint256).max) allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}
