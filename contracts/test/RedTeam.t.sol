// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {ERC1967Utils} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Utils.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts/proxy/utils/UUPSUpgradeable.sol";
import {Initializable} from "@openzeppelin/contracts/proxy/utils/Initializable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {SafeCast} from "@openzeppelin/contracts/utils/math/SafeCast.sol";
import {IAccessControl} from "@openzeppelin/contracts/access/IAccessControl.sol";
import {TimelockController} from "@openzeppelin/contracts/governance/TimelockController.sol";
import {PausableUpgradeable} from "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";
import {FlyVault} from "../src/FlyVault.sol";
import {FlyVaultStack} from "../script/FlyVaultStack.sol";
import {FlyVaultV2} from "./mocks/FlyVaultV2.sol";
import {ReentrantToken} from "./mocks/ReentrantToken.sol";
import {VaultTest} from "./utils/VaultTest.sol";

/// @notice Adversarial scenarios against FlyVault, its proxy and its timelock. Each test is an attack that must fail,
///         or a trust boundary that must hold exactly as documented (contracts/README.md, docs/vault/SPEC.md §1).
contract RedTeamTest is VaultTest {
    uint8 constant PENDING = 1;
    uint8 constant CANCELLED = 2;
    uint8 constant WITHDRAWN = 3;

    // ------------------------------------------------------------------ reentrancy into the unguarded functions

    /// A token hook that re-enters `cancelRequest` while `withdraw` is paying out gains nothing: the withdrawn request
    /// is already closed, the cancelled one just re-locks, and the vault stays exactly solvent.
    function test_redteam_reenterCancelDuringWithdrawGainsNothing() public {
        (FlyVault v, ReentrantToken t, Attacker a) = _reentrantSetup(10 ether);
        uint256 x = a.request(4 ether);
        uint256 y = a.request(6 ether);
        skip(WITHDRAW_DELAY);
        t.arm(address(a), abi.encodeCall(Attacker.reenterCancel, (y)));
        a.withdraw(x);

        assertEq(t.balanceOf(address(a)), 4 ether);
        assertEq(v.locked(address(a)), 6 ether);
        assertEq(v.pendingOf(address(a)), 0);
        assertEq(t.balanceOf(address(v)), v.totalLocked() + v.totalPending());
        (,,, uint8 sx) = v.request(x);
        (,,, uint8 sy) = v.request(y);
        assertEq(sx, WITHDRAWN);
        assertEq(sy, CANCELLED);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.RequestNotPending.selector, y));
        a.withdraw(y);
    }

    /// Re-entering `requestWithdrawal` mid-`withdraw` only creates an ordinary new request from the remaining balance.
    function test_redteam_reenterRequestDuringWithdrawIsJustANewRequest() public {
        (FlyVault v, ReentrantToken t, Attacker a) = _reentrantSetup(10 ether);
        uint256 x = a.request(4 ether);
        skip(WITHDRAW_DELAY);
        t.arm(address(a), abi.encodeCall(Attacker.reenterRequest, (6 ether)));
        a.withdraw(x);

        assertEq(t.balanceOf(address(a)), 4 ether);
        assertEq(v.locked(address(a)), 0);
        assertEq(v.pendingOf(address(a)), 6 ether);
        assertEq(t.balanceOf(address(v)), 6 ether);
        assertEq(t.balanceOf(address(v)), v.totalLocked() + v.totalPending());
    }

    /// `lock` inside a `withdraw` hook hits the shared guard.
    function test_redteam_reenterLockDuringWithdrawReverts() public {
        (, ReentrantToken t, Attacker a) = _reentrantSetup(10 ether);
        uint256 x = a.request(4 ether);
        skip(WITHDRAW_DELAY);
        t.arm(address(a), abi.encodeCall(Attacker.reenterLock, (1 ether)));
        vm.expectRevert(ReentrancyGuard.ReentrancyGuardReentrantCall.selector);
        a.withdraw(x);
    }

    /// `cancelRequest` inside a `lock` hook re-locks a pending request before the lock is credited; the lock then credits
    /// exactly what arrived, so nothing is double counted.
    function test_redteam_reenterCancelDuringLockStaysConsistent() public {
        (FlyVault v, ReentrantToken t, Attacker a) = _reentrantSetup(10 ether);
        uint256 x = a.request(5 ether);
        t.mint(address(a), 1 ether);
        t.arm(address(a), abi.encodeCall(Attacker.reenterCancel, (x)));
        a.lock(1 ether);

        assertEq(v.locked(address(a)), 11 ether);
        assertEq(v.pendingOf(address(a)), 0);
        assertEq(t.balanceOf(address(v)), 11 ether);
        assertEq(t.balanceOf(address(v)), v.totalLocked() + v.totalPending());
    }

    /// A hook that pushes extra tokens into the vault mid-`lock` makes `received != amount`, so the lock is refused
    /// rather than crediting the wrong figure.
    function test_redteam_donationInsideLockHookIsRefused() public {
        (FlyVault v, ReentrantToken t, Attacker a) = _reentrantSetup(0);
        t.mint(address(a), 3 ether);
        t.arm(address(a), abi.encodeCall(Attacker.donate, (1 ether)));
        vm.expectRevert(abi.encodeWithSelector(FlyVault.TransferAmountMismatch.selector, 2 ether, 3 ether));
        a.lock(2 ether);
        assertEq(v.totalLocked(), 0);
    }

    // ------------------------------------------------------------------ token behaviours

    function test_redteam_tokenReturningFalseCannotLock() public {
        FalseToken f = new FalseToken();
        FlyVault v = _vaultFor(address(f));
        f.mint(alice, 1 ether);
        vm.startPrank(alice);
        f.approve(address(v), 1 ether);
        vm.expectRevert(abi.encodeWithSelector(SafeERC20.SafeERC20FailedOperation.selector, address(f)));
        v.lock(1 ether);
        vm.stopPrank();
        assertEq(v.totalLocked(), 0);
    }

    /// A USDT-style token with no return values goes through SafeERC20 on both legs.
    function test_redteam_noReturnTokenLocksAndWithdraws() public {
        NoReturnToken n = new NoReturnToken();
        FlyVault v = _vaultFor(address(n));
        n.mint(alice, 5 ether);
        vm.startPrank(alice);
        n.approve(address(v), 5 ether);
        v.lock(5 ether);
        uint256 id = v.requestWithdrawal(5 ether);
        skip(WITHDRAW_DELAY);
        v.withdraw(id);
        vm.stopPrank();
        assertEq(n.balanceOf(alice), 5 ether);
        assertEq(n.balanceOf(address(v)), 0);
    }

    /// Tokens sent straight to the vault belong to nobody: no balance moves, and no holder can withdraw more than
    /// they locked, so the donation is simply stuck (there is no sweep).
    function test_redteam_directDonationIsNeverCredited() public {
        _lock(alice, 10 ether);
        fly.mint(eve, 3 ether);
        vm.prank(eve);
        fly.transfer(address(vault), 3 ether);

        assertEq(vault.locked(eve), 0);
        assertEq(vault.totalLocked(), 10 ether);
        vm.prank(eve);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.InsufficientLocked.selector, 1, 0));
        vault.requestWithdrawal(1);

        uint256 id = _request(alice, 10 ether);
        skip(WITHDRAW_DELAY);
        vm.prank(alice);
        vault.withdraw(id);
        assertEq(fly.balanceOf(alice), 10 ether);
        assertEq(fly.balanceOf(address(vault)), 3 ether);
        assertEq(vault.totalLocked() + vault.totalPending(), 0);
    }

    // ------------------------------------------------------------------ request ownership and ids

    function test_redteam_requestsCannotBeForgedStolenOrReplayed() public {
        _lock(alice, 5 ether);
        uint256 id = _request(alice, 5 ether);
        skip(WITHDRAW_DELAY);

        vm.startPrank(eve);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.NotRequestOwner.selector, id));
        vault.withdraw(id);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.NotRequestOwner.selector, id));
        vault.cancelRequest(id);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.UnknownRequest.selector, id + 1));
        vault.withdraw(id + 1);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.UnknownRequest.selector, type(uint256).max));
        vault.withdraw(type(uint256).max);
        vm.stopPrank();

        vm.startPrank(alice);
        vault.withdraw(id);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.RequestNotPending.selector, id));
        vault.withdraw(id);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.RequestNotPending.selector, id));
        vault.cancelRequest(id);
        vm.stopPrank();
        assertEq(fly.balanceOf(alice), 5 ether);
    }

    /// Any split of a locked balance into partial requests withdraws exactly the requested total, never more.
    function testFuzz_redteam_partialRequestsWithdrawExactly(uint96 total, uint256 seed) public {
        total = uint96(bound(total, 1, fly.MAX_MINT()));
        _lock(alice, total);
        uint256 remaining = total;
        uint256 requested;
        uint256[] memory ids = new uint256[](5);
        uint256 n;
        for (uint256 i; i < 5 && remaining > 0; ++i) {
            uint256 amt = bound(uint256(keccak256(abi.encode(seed, i))), 1, remaining);
            ids[n++] = _request(alice, amt);
            requested += amt;
            remaining -= amt;
        }
        vm.prank(alice);
        vm.expectRevert();
        vault.requestWithdrawal(remaining + 1);
        skip(WITHDRAW_DELAY);
        vm.startPrank(alice);
        for (uint256 i; i < n; ++i) {
            vault.withdraw(ids[i]);
        }
        vm.stopPrank();
        assertEq(fly.balanceOf(alice), requested);
        assertEq(vault.locked(alice), remaining);
        assertEq(vault.pendingOf(alice), 0);
        assertEq(fly.balanceOf(address(vault)), remaining);
    }

    // ------------------------------------------------------------------ pauser limits

    function test_redteam_pauserHasNoOtherPower() public {
        bytes32 upgrader = vault.UPGRADER_ROLE();
        bytes32 admin = vault.DEFAULT_ADMIN_ROLE();
        FlyVaultV2 v2 = new FlyVaultV2(WITHDRAW_DELAY);
        vm.startPrank(pauser);
        vm.expectRevert(abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, pauser, upgrader));
        vault.upgradeToAndCall(address(v2), "");
        vm.expectRevert(abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, pauser, admin));
        vault.grantRole(upgrader, pauser);
        vm.expectRevert(abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, pauser, admin));
        vault.revokeRole(admin, address(timelock));
        vm.expectRevert(
            abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, pauser, timelock.PROPOSER_ROLE())
        );
        timelock.schedule(address(vault), 0, "", bytes32(0), bytes32(0), TIMELOCK_DELAY);
        vm.stopPrank();
    }

    /// A pause that never ends still lets every holder take everything out.
    function testFuzz_redteam_permanentPauseCannotTrapFunds(uint96 a, uint96 b, uint96 c) public {
        a = uint96(bound(a, 1, fly.MAX_MINT()));
        b = uint96(bound(b, 1, fly.MAX_MINT()));
        c = uint96(bound(c, 2, fly.MAX_MINT())); // split into two requests below
        _lock(alice, a);
        _lock(bob, b);
        _lock(eve, c);
        uint256 pendingBefore = _request(eve, c / 2); // a request that was pending when the pause hit

        vm.prank(pauser);
        vault.pause();

        vm.prank(eve);
        vm.expectRevert(PausableUpgradeable.EnforcedPause.selector);
        vault.cancelRequest(pendingBefore);

        uint256 ia = _request(alice, a);
        uint256 ib = _request(bob, b);
        uint256 ic = _request(eve, c - c / 2);
        skip(WITHDRAW_DELAY);
        vm.prank(alice);
        vault.withdraw(ia);
        vm.prank(bob);
        vault.withdraw(ib);
        vm.startPrank(eve);
        vault.withdraw(pendingBefore);
        vault.withdraw(ic);
        vm.stopPrank();

        assertEq(fly.balanceOf(alice), a);
        assertEq(fly.balanceOf(bob), b);
        assertEq(fly.balanceOf(eve), c);
        assertEq(fly.balanceOf(address(vault)), 0);
        assertTrue(vault.paused());
    }

    // ------------------------------------------------------------------ governance: the timelock is the only path

    /// The mainnet parameters guarantee an exit window: every upgrade waits longer than any withdrawal.
    function test_redteam_mainnetDelaysLeaveAnExitWindow() public pure {
        assertGt(FlyVaultStack.MAINNET_TIMELOCK_DELAY, FlyVaultStack.MAINNET_WITHDRAW_DELAY);
        assertEq(FlyVaultStack.MAINNET_TIMELOCK_DELAY - FlyVaultStack.MAINNET_WITHDRAW_DELAY, 1 days);
    }

    /// A malicious implementation can drain the vault, but only after the full timelock delay; a holder who requests a
    /// withdrawal up to a day after the schedule is out before it can execute.
    function testFuzz_redteam_holdersCanExitBeforeAMaliciousUpgradeExecutes(uint32 lag, uint96 amt) public {
        amt = uint96(bound(amt, 1, fly.MAX_MINT()));
        lag = uint32(bound(lag, 0, TIMELOCK_DELAY - WITHDRAW_DELAY - 1));
        _lock(alice, amt);
        _lock(bob, 1 ether); // stays in: what the upgrade can still take

        EvilVault evil = new EvilVault();
        bytes memory data = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (address(evil), ""));
        uint256 t0 = block.timestamp;
        vm.prank(owner);
        timelock.schedule(address(vault), 0, data, bytes32(0), bytes32(0), TIMELOCK_DELAY);

        skip(lag);
        uint256 id = _request(alice, amt);
        vm.warp(t0 + lag + WITHDRAW_DELAY);
        assertLt(block.timestamp, t0 + TIMELOCK_DELAY);
        vm.expectRevert();
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));
        vm.prank(alice);
        vault.withdraw(id);
        assertEq(fly.balanceOf(alice), amt);

        vm.warp(t0 + TIMELOCK_DELAY);
        vm.prank(eve);
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));
        EvilVault(address(vault)).sweep(eve);
        assertEq(fly.balanceOf(eve), 1 ether); // only bob's tokens were still there
        assertEq(fly.balanceOf(alice), amt);
    }

    /// Nobody, the owner included, can execute anything but the exact scheduled call.
    function test_redteam_executorCannotAlterTheScheduledCall() public {
        FlyVaultV2 v2 = new FlyVaultV2(WITHDRAW_DELAY);
        EvilVault evil = new EvilVault();
        bytes memory good = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (address(v2), ""));
        bytes memory bad = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (address(evil), ""));
        vm.prank(owner);
        timelock.schedule(address(vault), 0, good, bytes32(0), bytes32(0), TIMELOCK_DELAY);
        skip(TIMELOCK_DELAY);
        bytes32 badId = timelock.hashOperation(address(vault), 0, bad, bytes32(0), bytes32(0));
        vm.prank(owner);
        vm.expectRevert(
            abi.encodeWithSelector(TimelockController.TimelockUnexpectedOperationState.selector, badId, bytes32(uint256(1 << 2)))
        );
        timelock.execute(address(vault), 0, bad, bytes32(0), bytes32(0));
        vm.prank(eve);
        timelock.execute(address(vault), 0, good, bytes32(0), bytes32(0));
        assertEq(_implementation(), address(v2));
    }

    /// An upgrade cannot smuggle a second `initialize` in its call data to reassign the token or roles.
    function test_redteam_upgradeCannotReinitialize() public {
        FlyVaultV2 v2 = new FlyVaultV2(WITHDRAW_DELAY);
        bytes memory reinit = abi.encodeCall(FlyVault.initialize, (address(0xBEEF), eve, eve));
        bytes memory data = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (address(v2), reinit));
        vm.prank(owner);
        timelock.schedule(address(vault), 0, data, bytes32(0), bytes32(0), TIMELOCK_DELAY);
        skip(TIMELOCK_DELAY);
        vm.expectRevert(Initializable.InvalidInitialization.selector);
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));
        assertEq(vault.token(), address(fly));
        assertFalse(vault.hasRole(vault.DEFAULT_ADMIN_ROLE(), eve));
    }

    /// Trust boundary, documented: the timelock (as DEFAULT_ADMIN) can hand UPGRADER_ROLE to an EOA. That grant is
    /// itself a public 8-day operation, but once it lands the EOA upgrades with no delay at all. Holders must treat
    /// a scheduled `grantRole(UPGRADER_ROLE, …)` exactly like a scheduled upgrade.
    function test_redteam_grantingUpgraderToAnEoaRemovesTheDelayAfterwards() public {
        bytes32 upgrader = vault.UPGRADER_ROLE();
        bytes memory data = abi.encodeCall(IAccessControl.grantRole, (upgrader, owner));
        vm.prank(owner);
        timelock.schedule(address(vault), 0, data, bytes32(0), bytes32(0), TIMELOCK_DELAY);
        skip(TIMELOCK_DELAY - 1);
        vm.expectRevert();
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));
        skip(1);
        timelock.execute(address(vault), 0, data, bytes32(0), bytes32(0));

        EvilVault evil = new EvilVault();
        vm.prank(owner);
        vault.upgradeToAndCall(address(evil), ""); // instant: no timelock any more
        assertEq(_implementation(), address(evil));
    }

    /// Even an account that becomes the timelock's admin (an 8-day operation) cannot shorten or skip the delay.
    function test_redteam_timelockAdminStillCannotSkipTheDelay() public {
        bytes32 tAdmin = timelock.DEFAULT_ADMIN_ROLE();
        bytes memory data = abi.encodeCall(IAccessControl.grantRole, (tAdmin, owner));
        vm.prank(owner);
        timelock.schedule(address(timelock), 0, data, bytes32(0), bytes32(0), TIMELOCK_DELAY);
        skip(TIMELOCK_DELAY);
        timelock.execute(address(timelock), 0, data, bytes32(0), bytes32(0));
        assertTrue(timelock.hasRole(tAdmin, owner));

        vm.startPrank(owner);
        timelock.grantRole(timelock.PROPOSER_ROLE(), eve); // instant, but harmless on its own
        vm.expectRevert(abi.encodeWithSelector(TimelockController.TimelockUnauthorizedCaller.selector, owner));
        timelock.updateDelay(0);
        vm.stopPrank();

        FlyVaultV2 v2 = new FlyVaultV2(WITHDRAW_DELAY);
        bytes memory up = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (address(v2), ""));
        vm.prank(eve);
        vm.expectRevert(abi.encodeWithSelector(TimelockController.TimelockInsufficientDelay.selector, 0, TIMELOCK_DELAY));
        timelock.schedule(address(vault), 0, up, bytes32(0), bytes32(0), 0);
    }

    // ------------------------------------------------------------------ structure

    /// Every piece of state lives in an ERC-7201 namespace: the proxy's low slots stay untouched through a full
    /// lifecycle, so no inherited contract added a plain state variable that an upgrade could collide with.
    function test_redteam_noPlainStorageSlotsAreEverWritten() public {
        _lock(alice, 5 ether);
        uint256 id = _request(alice, 2 ether);
        vm.prank(alice);
        vault.cancelRequest(id);
        uint256 id2 = _request(alice, 1 ether);
        skip(WITHDRAW_DELAY);
        vm.prank(alice);
        vault.withdraw(id2);
        vm.prank(pauser);
        vault.pause();
        for (uint256 i; i < 64; ++i) {
            assertEq(vm.load(address(vault), bytes32(i)), bytes32(0), "plain slot written");
            assertEq(vm.load(address(implementation), bytes32(i)), bytes32(0), "plain slot written (impl)");
        }
    }

    /// A withdraw delay that overflows uint64 makes every request revert instead of producing a bogus readyAt.
    function test_redteam_absurdWithdrawDelayRevertsAtRequest() public {
        FlyVault impl = new FlyVault(type(uint64).max);
        bytes memory init = abi.encodeCall(FlyVault.initialize, (address(fly), pauser, address(timelock)));
        FlyVault v = FlyVault(address(new ERC1967Proxy(address(impl), init)));
        fly.mint(alice, 1 ether);
        vm.startPrank(alice);
        fly.approve(address(v), 1 ether);
        v.lock(1 ether);
        vm.expectRevert(
            abi.encodeWithSelector(SafeCast.SafeCastOverflowedUintDowncast.selector, 64, block.timestamp + type(uint64).max)
        );
        v.requestWithdrawal(1 ether);
        vm.stopPrank();
    }

    /// The implementation contract is inert on its own: nobody can initialize it, upgrade it, or use it as a vault.
    function test_redteam_implementationIsInert() public {
        vm.expectRevert(Initializable.InvalidInitialization.selector);
        implementation.initialize(address(fly), eve, eve);
        vm.prank(address(timelock));
        vm.expectRevert(UUPSUpgradeable.UUPSUnauthorizedCallContext.selector);
        implementation.upgradeToAndCall(address(0xBEEF), "");
        vm.expectRevert(); // token() is address(0): lock cannot even call balanceOf
        implementation.lock(1);
        assertEq(implementation.token(), address(0));
    }

    // ------------------------------------------------------------------ helpers

    function _implementation() internal view returns (address) {
        return address(uint160(uint256(vm.load(address(vault), ERC1967Utils.IMPLEMENTATION_SLOT))));
    }

    function _vaultFor(address token_) internal returns (FlyVault) {
        bytes memory init = abi.encodeCall(FlyVault.initialize, (token_, pauser, address(timelock)));
        return FlyVault(address(new ERC1967Proxy(address(implementation), init)));
    }

    function _reentrantSetup(uint256 locked_) internal returns (FlyVault v, ReentrantToken t, Attacker a) {
        t = new ReentrantToken();
        v = _vaultFor(address(t));
        a = new Attacker(v, t);
        if (locked_ > 0) {
            t.mint(address(a), locked_);
            a.lock(locked_);
        }
    }
}

/// @dev Holder contract whose token hook re-enters the vault through the functions that carry no reentrancy guard.
contract Attacker {
    FlyVault internal immutable vault;
    ReentrantToken internal immutable token;

    constructor(FlyVault v, ReentrantToken t) {
        vault = v;
        token = t;
        t.approve(address(v), type(uint256).max);
    }

    function lock(uint256 amount) external {
        vault.lock(amount);
    }

    function request(uint256 amount) external returns (uint256) {
        return vault.requestWithdrawal(amount);
    }

    function withdraw(uint256 id) external {
        vault.withdraw(id);
    }

    function reenterCancel(uint256 id) external {
        vault.cancelRequest(id);
    }

    function reenterRequest(uint256 amount) external {
        vault.requestWithdrawal(amount);
    }

    function reenterLock(uint256 amount) external {
        vault.lock(amount);
    }

    function donate(uint256 amount) external {
        token.transfer(address(vault), amount);
    }
}

/// @dev Upgrade target an attacker who controls the upgrader would deploy.
contract EvilVault is FlyVault {
    constructor() FlyVault(7 days) {}

    function sweep(address to) external {
        IERC20 t = IERC20(this.token());
        t.transfer(to, t.balanceOf(address(this)));
    }
}

/// @dev ERC-20 whose transfers return false instead of reverting.
contract FalseToken {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transfer(address, uint256) external pure returns (bool) {
        return false;
    }

    function transferFrom(address, address, uint256) external pure returns (bool) {
        return false;
    }
}

/// @dev USDT-style ERC-20: no return values at all.
contract NoReturnToken {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
    }

    function approve(address spender, uint256 amount) external {
        allowance[msg.sender][spender] = amount;
    }

    function transfer(address to, uint256 amount) external {
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
    }

    function transferFrom(address from, address to, uint256 amount) external {
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
    }
}
