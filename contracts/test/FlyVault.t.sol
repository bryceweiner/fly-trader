// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IAccessControl} from "@openzeppelin/contracts/access/IAccessControl.sol";
import {IERC20Errors} from "@openzeppelin/contracts/interfaces/draft-IERC6093.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {PausableUpgradeable} from "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";
import {FlyVault} from "../src/FlyVault.sol";
import {FeeToken} from "./mocks/FeeToken.sol";
import {ReentrantToken} from "./mocks/ReentrantToken.sol";
import {VaultTest} from "./utils/VaultTest.sol";

contract FlyVaultTest is VaultTest {
    event Locked(address indexed user, uint256 amount, uint256 lockedAfter);
    event WithdrawRequested(
        address indexed user, uint256 indexed id, uint256 amount, uint64 readyAt, uint256 lockedAfter
    );
    event RequestCancelled(address indexed user, uint256 indexed id, uint256 amount, uint256 lockedAfter);
    event Withdrawn(address indexed user, uint256 indexed id, uint256 amount);

    uint8 constant NONE = 0;
    uint8 constant PENDING = 1;
    uint8 constant CANCELLED = 2;
    uint8 constant WITHDRAWN = 3;

    // ------------------------------------------------------------------ setup / init

    function test_initialState() public view {
        assertEq(vault.token(), address(fly));
        assertEq(vault.WITHDRAW_DELAY(), WITHDRAW_DELAY);
        assertEq(implementation.WITHDRAW_DELAY(), WITHDRAW_DELAY);
        assertEq(vault.PAUSER_ROLE(), keccak256("PAUSER_ROLE"));
        assertEq(vault.UPGRADER_ROLE(), keccak256("UPGRADER_ROLE"));
        assertTrue(vault.hasRole(vault.DEFAULT_ADMIN_ROLE(), address(timelock)));
        assertTrue(vault.hasRole(vault.UPGRADER_ROLE(), address(timelock)));
        assertTrue(vault.hasRole(vault.PAUSER_ROLE(), pauser));
        assertFalse(vault.hasRole(vault.PAUSER_ROLE(), address(timelock)));
        assertEq(vault.totalLocked(), 0);
        assertEq(vault.totalPending(), 0);
        assertFalse(vault.paused());
    }

    function test_storageSlotIsErc7201() public view {
        bytes32 expected =
            keccak256(abi.encode(uint256(keccak256("fly.storage.FlyVault")) - 1)) & ~bytes32(uint256(0xff));
        // token is the first field of the namespaced struct
        assertEq(address(uint160(uint256(vm.load(address(vault), expected)))), address(fly));
    }

    function test_initialize_rejectsZeroAddresses() public {
        FlyVault impl = new FlyVault(WITHDRAW_DELAY);
        address t = address(timelock);
        address[3][3] memory args = [[address(0), pauser, t], [address(fly), address(0), t], [address(fly), pauser, address(0)]];
        for (uint256 i; i < 3; ++i) {
            bytes memory init = abi.encodeCall(FlyVault.initialize, (args[i][0], args[i][1], args[i][2]));
            vm.expectRevert(FlyVault.ZeroAddress.selector);
            new ERC1967Proxy(address(impl), init);
        }
    }

    function test_proxyRequiresInitData() public {
        FlyVault impl = new FlyVault(WITHDRAW_DELAY);
        vm.expectRevert(ERC1967Proxy.ERC1967ProxyUninitialized.selector);
        new ERC1967Proxy(address(impl), "");
    }

    // ------------------------------------------------------------------ lock

    function test_lock() public {
        _fund(alice, 100 ether);
        vm.expectEmit(address(vault));
        emit Locked(alice, 60 ether, 60 ether);
        vm.prank(alice);
        vault.lock(60 ether);

        vm.expectEmit(address(vault));
        emit Locked(alice, 40 ether, 100 ether);
        vm.prank(alice);
        vault.lock(40 ether);

        assertEq(vault.locked(alice), 100 ether);
        assertEq(vault.totalLocked(), 100 ether);
        assertEq(fly.balanceOf(address(vault)), 100 ether);
        assertEq(fly.balanceOf(alice), 0);
    }

    function test_lock_zeroReverts() public {
        _fund(alice, 1 ether);
        vm.prank(alice);
        vm.expectRevert(FlyVault.ZeroAmount.selector);
        vault.lock(0);
    }

    function test_lock_withoutAllowanceReverts() public {
        fly.mint(alice, 1 ether);
        vm.prank(alice);
        vm.expectRevert(
            abi.encodeWithSelector(IERC20Errors.ERC20InsufficientAllowance.selector, address(vault), 0, 1 ether)
        );
        vault.lock(1 ether);
    }

    function test_lock_moreThanBalanceReverts() public {
        _fund(alice, 1 ether);
        vm.prank(alice);
        vm.expectRevert(
            abi.encodeWithSelector(IERC20Errors.ERC20InsufficientBalance.selector, alice, 1 ether, 2 ether)
        );
        vault.lock(2 ether);
    }

    function test_lock_feeOnTransferReverts() public {
        FeeToken fee = new FeeToken();
        FlyVault v = _vaultFor(address(fee));
        fee.mint(alice, 100 ether);
        vm.startPrank(alice);
        fee.approve(address(v), type(uint256).max);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.TransferAmountMismatch.selector, 100 ether, 99 ether));
        v.lock(100 ether);
        vm.stopPrank();
        assertEq(v.totalLocked(), 0);
        assertEq(fee.balanceOf(alice), 100 ether);
    }

    // ------------------------------------------------------------------ requestWithdrawal

    function test_requestWithdrawal() public {
        _lock(alice, 100 ether);
        uint64 readyAt = uint64(block.timestamp + WITHDRAW_DELAY);
        vm.expectEmit(address(vault));
        emit WithdrawRequested(alice, 1, 30 ether, readyAt, 70 ether);
        uint256 id = _request(alice, 30 ether);

        assertEq(id, 1);
        assertEq(vault.locked(alice), 70 ether);
        assertEq(vault.pendingOf(alice), 30 ether);
        assertEq(vault.totalLocked(), 70 ether);
        assertEq(vault.totalPending(), 30 ether);
        (address user, uint256 amount, uint64 r, uint8 state) = vault.request(id);
        assertEq(user, alice);
        assertEq(amount, 30 ether);
        assertEq(r, readyAt);
        assertEq(state, PENDING);
        assertEq(fly.balanceOf(address(vault)), 100 ether); // nothing leaves yet
    }

    function test_requestWithdrawal_all() public {
        _lock(alice, 5 ether);
        vm.expectEmit(address(vault));
        emit WithdrawRequested(alice, 1, 5 ether, uint64(block.timestamp + WITHDRAW_DELAY), 0);
        _request(alice, 5 ether);
        assertEq(vault.locked(alice), 0);
    }

    function test_requestWithdrawal_idsAreGlobalAndSequential() public {
        _lock(alice, 10 ether);
        _lock(bob, 10 ether);
        assertEq(_request(alice, 1 ether), 1);
        assertEq(_request(bob, 1 ether), 2);
        assertEq(_request(alice, 1 ether), 3);
        uint256[] memory a = vault.requestsOf(alice);
        uint256[] memory b = vault.requestsOf(bob);
        assertEq(a.length, 2);
        assertEq(a[0], 1);
        assertEq(a[1], 3);
        assertEq(b.length, 1);
        assertEq(b[0], 2);
        assertEq(vault.requestsOf(eve).length, 0);
    }

    function test_requestWithdrawal_zeroReverts() public {
        _lock(alice, 1 ether);
        vm.prank(alice);
        vm.expectRevert(FlyVault.ZeroAmount.selector);
        vault.requestWithdrawal(0);
    }

    function test_requestWithdrawal_moreThanLockedReverts() public {
        _lock(alice, 1 ether);
        vm.prank(alice);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.InsufficientLocked.selector, 1 ether + 1, 1 ether));
        vault.requestWithdrawal(1 ether + 1);

        // pending tokens are not locked any more
        _request(alice, 1 ether);
        vm.prank(alice);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.InsufficientLocked.selector, 1, 0));
        vault.requestWithdrawal(1);
    }

    function test_request_unknownIdIsEmpty() public view {
        (address user, uint256 amount, uint64 readyAt, uint8 state) = vault.request(42);
        assertEq(user, address(0));
        assertEq(amount, 0);
        assertEq(readyAt, 0);
        assertEq(state, NONE);
    }

    // ------------------------------------------------------------------ cancelRequest

    function test_cancelRequest() public {
        _lock(alice, 100 ether);
        uint256 id = _request(alice, 40 ether);
        vm.expectEmit(address(vault));
        emit RequestCancelled(alice, id, 40 ether, 100 ether);
        vm.prank(alice);
        vault.cancelRequest(id);

        assertEq(vault.locked(alice), 100 ether);
        assertEq(vault.pendingOf(alice), 0);
        assertEq(vault.totalLocked(), 100 ether);
        assertEq(vault.totalPending(), 0);
        (,,, uint8 state) = vault.request(id);
        assertEq(state, CANCELLED);
    }

    function test_cancelRequest_foreignUserReverts() public {
        _lock(alice, 1 ether);
        uint256 id = _request(alice, 1 ether);
        vm.prank(eve);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.NotRequestOwner.selector, id));
        vault.cancelRequest(id);
    }

    function test_cancelRequest_unknownIdReverts() public {
        _lock(alice, 1 ether);
        _request(alice, 1 ether);
        vm.startPrank(alice);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.UnknownRequest.selector, 0));
        vault.cancelRequest(0);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.UnknownRequest.selector, 2));
        vault.cancelRequest(2);
        vm.stopPrank();
    }

    function test_cancelRequest_twiceReverts() public {
        _lock(alice, 1 ether);
        uint256 id = _request(alice, 1 ether);
        vm.startPrank(alice);
        vault.cancelRequest(id);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.RequestNotPending.selector, id));
        vault.cancelRequest(id);
        vm.stopPrank();
    }

    function test_cancelRequest_withdrawnReverts() public {
        _lock(alice, 1 ether);
        uint256 id = _request(alice, 1 ether);
        skip(WITHDRAW_DELAY);
        vm.startPrank(alice);
        vault.withdraw(id);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.RequestNotPending.selector, id));
        vault.cancelRequest(id);
        vm.stopPrank();
    }

    // ------------------------------------------------------------------ withdraw

    function test_withdraw_readyAtBoundary() public {
        _lock(alice, 100 ether);
        uint256 requestedAt = block.timestamp;
        uint256 id = _request(alice, 25 ether);
        uint64 readyAt = uint64(requestedAt + WITHDRAW_DELAY);

        vm.warp(readyAt - 1);
        vm.prank(alice);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.WithdrawalNotReady.selector, id, readyAt));
        vault.withdraw(id);

        vm.warp(readyAt); // exactly WITHDRAW_DELAY after the request
        vm.expectEmit(address(vault));
        emit Withdrawn(alice, id, 25 ether);
        vm.prank(alice);
        vault.withdraw(id);

        assertEq(fly.balanceOf(alice), 25 ether);
        assertEq(fly.balanceOf(address(vault)), 75 ether);
        assertEq(vault.pendingOf(alice), 0);
        assertEq(vault.totalPending(), 0);
        assertEq(vault.locked(alice), 75 ether);
        assertEq(vault.totalLocked(), 75 ether);
        (,,, uint8 state) = vault.request(id);
        assertEq(state, WITHDRAWN);
    }

    function test_withdraw_twiceReverts() public {
        _lock(alice, 1 ether);
        uint256 id = _request(alice, 1 ether);
        skip(WITHDRAW_DELAY);
        vm.startPrank(alice);
        vault.withdraw(id);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.RequestNotPending.selector, id));
        vault.withdraw(id);
        vm.stopPrank();
    }

    function test_withdraw_cancelledReverts() public {
        _lock(alice, 1 ether);
        uint256 id = _request(alice, 1 ether);
        vm.prank(alice);
        vault.cancelRequest(id);
        skip(WITHDRAW_DELAY);
        vm.prank(alice);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.RequestNotPending.selector, id));
        vault.withdraw(id);
    }

    function test_withdraw_foreignUserReverts() public {
        _lock(alice, 1 ether);
        uint256 id = _request(alice, 1 ether);
        skip(WITHDRAW_DELAY);
        vm.prank(eve);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.NotRequestOwner.selector, id));
        vault.withdraw(id);
    }

    function test_withdraw_unknownIdReverts() public {
        vm.prank(alice);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.UnknownRequest.selector, 7));
        vault.withdraw(7);
    }

    function test_withdraw_multipleRequestsIndependent() public {
        _lock(alice, 10 ether);
        uint256 a = _request(alice, 3 ether);
        skip(1 days);
        uint256 b = _request(alice, 4 ether);
        skip(WITHDRAW_DELAY - 1 days);
        vm.startPrank(alice);
        vault.withdraw(a);
        vm.expectRevert(
            abi.encodeWithSelector(FlyVault.WithdrawalNotReady.selector, b, uint64(block.timestamp + 1 days))
        );
        vault.withdraw(b);
        skip(1 days);
        vault.withdraw(b);
        vm.stopPrank();
        assertEq(fly.balanceOf(alice), 7 ether);
        assertEq(vault.locked(alice), 3 ether);
    }

    // ------------------------------------------------------------------ pause scope

    function test_pause_onlyPauser() public {
        bytes32 role = vault.PAUSER_ROLE();
        address[3] memory others = [owner, eve, address(timelock)];
        for (uint256 i; i < others.length; ++i) {
            vm.prank(others[i]);
            vm.expectRevert(
                abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, others[i], role)
            );
            vault.pause();
        }
        vm.prank(pauser);
        vault.pause();
        vm.prank(eve);
        vm.expectRevert(abi.encodeWithSelector(IAccessControl.AccessControlUnauthorizedAccount.selector, eve, role));
        vault.unpause();
    }

    function test_pause_blocksLockAndCancelOnly() public {
        _lock(alice, 100 ether);
        uint256 early = _request(alice, 10 ether);
        _fund(bob, 1 ether);
        skip(WITHDRAW_DELAY);

        vm.prank(pauser);
        vault.pause();
        assertTrue(vault.paused());

        vm.prank(bob);
        vm.expectRevert(PausableUpgradeable.EnforcedPause.selector);
        vault.lock(1 ether);

        // leaving always works
        uint256 late = _request(alice, 20 ether);
        vm.prank(alice);
        vault.withdraw(early);
        assertEq(fly.balanceOf(alice), 10 ether);

        vm.prank(alice);
        vm.expectRevert(PausableUpgradeable.EnforcedPause.selector);
        vault.cancelRequest(late);

        vm.prank(pauser);
        vault.unpause();
        vm.prank(alice);
        vault.cancelRequest(late);
        vm.prank(bob);
        vault.lock(1 ether);
        assertEq(vault.locked(alice), 90 ether);
        assertEq(vault.totalLocked(), 91 ether);
    }

    // ------------------------------------------------------------------ reentrancy

    function test_reentrancy_lockDuringLockReverts() public {
        ReentrantToken t = new ReentrantToken();
        FlyVault v = _vaultFor(address(t));
        address attacker = address(new Reenterer(v, t));
        t.mint(attacker, 10 ether);
        t.arm(attacker, abi.encodeCall(Reenterer.reenterLock, ()));
        vm.expectRevert(ReentrancyGuard.ReentrancyGuardReentrantCall.selector);
        Reenterer(attacker).lock(5 ether);
    }

    function test_reentrancy_withdrawDuringWithdrawReverts() public {
        ReentrantToken t = new ReentrantToken();
        FlyVault v = _vaultFor(address(t));
        Reenterer attacker = new Reenterer(v, t);
        t.mint(address(attacker), 10 ether);
        attacker.lock(10 ether);
        uint256 a = attacker.request(5 ether);
        uint256 b = attacker.request(5 ether);
        skip(WITHDRAW_DELAY);
        t.arm(address(attacker), abi.encodeCall(Reenterer.reenterWithdraw, (b)));
        vm.expectRevert(ReentrancyGuard.ReentrancyGuardReentrantCall.selector);
        attacker.withdraw(a);
    }

    function test_reentrancy_requestDuringLockSeesOldBalance() public {
        // A hook that re-enters the (unguarded) requestWithdrawal mid-lock sees the pre-lock balance: no double count.
        ReentrantToken t = new ReentrantToken();
        FlyVault v = _vaultFor(address(t));
        Reenterer attacker = new Reenterer(v, t);
        t.mint(address(attacker), 10 ether);
        t.arm(address(attacker), abi.encodeCall(Reenterer.reenterRequest, (1)));
        vm.expectRevert(abi.encodeWithSelector(FlyVault.InsufficientLocked.selector, 1, 0));
        attacker.lock(5 ether);
    }

    // ------------------------------------------------------------------ fuzz

    function testFuzz_lifecycle(uint96 lockAmt, uint96 reqAmt, bool cancel) public {
        lockAmt = uint96(bound(lockAmt, 1, fly.MAX_MINT()));
        reqAmt = uint96(bound(reqAmt, 1, lockAmt));
        _lock(alice, lockAmt);
        uint256 id = _request(alice, reqAmt);
        assertEq(vault.locked(alice) + vault.pendingOf(alice), lockAmt);
        if (cancel) {
            vm.prank(alice);
            vault.cancelRequest(id);
            assertEq(vault.locked(alice), lockAmt);
        } else {
            skip(WITHDRAW_DELAY);
            vm.prank(alice);
            vault.withdraw(id);
            assertEq(fly.balanceOf(alice), reqAmt);
            assertEq(vault.locked(alice), lockAmt - reqAmt);
        }
        assertEq(vault.pendingOf(alice), 0);
        assertEq(fly.balanceOf(address(vault)), vault.totalLocked() + vault.totalPending());
    }

    // ------------------------------------------------------------------ helpers

    function _vaultFor(address token_) internal returns (FlyVault) {
        bytes memory init = abi.encodeCall(FlyVault.initialize, (token_, pauser, address(timelock)));
        return FlyVault(address(new ERC1967Proxy(address(implementation), init)));
    }
}

/// @dev Holder contract whose token hook re-enters the vault.
contract Reenterer {
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

    function reenterLock() external {
        vault.lock(1);
    }

    function reenterWithdraw(uint256 id) external {
        vault.withdraw(id);
    }

    function reenterRequest(uint256 amount) external {
        vault.requestWithdrawal(amount);
    }
}
