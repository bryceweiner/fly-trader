// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {FlyVault} from "../src/FlyVault.sol";
import {MockFLY} from "../src/testnet/MockFLY.sol";
import {VaultTest} from "./utils/VaultTest.sol";

/// @notice Drives the vault with several actors, time warps, pauses and stray donations. Every call it makes is valid
///         for the current state (fail_on_revert = true), so an unexpected revert is a failure too.
contract VaultHandler is Test {
    FlyVault internal immutable vault;
    MockFLY internal immutable fly;
    address internal immutable pauser;
    uint256 internal immutable withdrawDelay;

    address[] public actors;
    uint256 public donated;
    uint256 public withdrawnTotal;
    mapping(address => uint256) public lockedBy; // everything each actor ever locked
    mapping(address => uint256) public withdrawnBy; // everything each actor ever got back
    mapping(bytes32 => uint256) public calls;

    constructor(FlyVault v, MockFLY f, address pauser_, uint256 withdrawDelay_) {
        (vault, fly, pauser, withdrawDelay) = (v, f, pauser_, withdrawDelay_);
        for (uint256 i; i < 4; ++i) {
            address a = makeAddr(string.concat("actor", vm.toString(i)));
            actors.push(a);
            vm.prank(a);
            fly.approve(address(vault), type(uint256).max);
        }
    }

    function actorCount() external view returns (uint256) {
        return actors.length;
    }

    function _actor(uint256 seed) internal view returns (address) {
        return actors[seed % actors.length];
    }

    function lock(uint256 actorSeed, uint256 amount) external {
        if (vault.paused()) return;
        address a = _actor(actorSeed);
        amount = bound(amount, 1, fly.MAX_MINT());
        fly.mint(a, amount);
        vm.prank(a);
        vault.lock(amount);
        lockedBy[a] += amount;
        calls["lock"]++;
    }

    function requestWithdrawal(uint256 actorSeed, uint256 amount) external {
        address a = _actor(actorSeed);
        uint256 available = vault.locked(a);
        if (available == 0) return;
        amount = bound(amount, 1, available);
        vm.prank(a);
        uint256 id = vault.requestWithdrawal(amount);
        (, , uint64 readyAt,) = vault.request(id);
        assertEq(readyAt, block.timestamp + withdrawDelay);
        calls["request"]++;
    }

    /// @dev Picks one of the actor's requests; only pending ones (and, for withdraw, ready ones) are acted on.
    function _pick(address a, uint256 seed) internal view returns (uint256 id, uint64 readyAt, bool pending) {
        uint256[] memory ids = vault.requestsOf(a);
        if (ids.length == 0) return (0, 0, false);
        id = ids[seed % ids.length];
        uint8 state;
        (, , readyAt, state) = vault.request(id);
        pending = state == 1;
    }

    function cancelRequest(uint256 actorSeed, uint256 idSeed) external {
        if (vault.paused()) return;
        address a = _actor(actorSeed);
        (uint256 id,, bool pending) = _pick(a, idSeed);
        if (!pending) return;
        vm.prank(a);
        vault.cancelRequest(id);
        calls["cancel"]++;
    }

    function withdraw(uint256 actorSeed, uint256 idSeed) external {
        address a = _actor(actorSeed);
        (uint256 id, uint64 readyAt, bool pending) = _pick(a, idSeed);
        if (!pending || block.timestamp < readyAt) return;
        (, uint256 amount,,) = vault.request(id);
        uint256 before = fly.balanceOf(a);
        vm.prank(a);
        vault.withdraw(id);
        assertEq(fly.balanceOf(a), before + amount);
        withdrawnTotal += amount;
        withdrawnBy[a] += amount;
        calls["withdraw"]++;
    }

    function warp(uint256 seconds_) external {
        skip(bound(seconds_, 1, 3 * withdrawDelay));
        calls["warp"]++;
    }

    function togglePause() external {
        bool paused = vault.paused();
        vm.prank(pauser);
        if (paused) vault.unpause();
        else vault.pause();
        calls["pause"]++;
    }

    /// @dev Tokens sent straight to the vault are not credited to anyone (and cannot be swept).
    function donate(uint256 amount) external {
        amount = bound(amount, 1, 1000 ether);
        fly.mint(address(vault), amount);
        donated += amount;
        calls["donate"]++;
    }
}

contract InvariantTest is VaultTest {
    VaultHandler internal handler;

    function setUp() public override {
        super.setUp();
        handler = new VaultHandler(vault, fly, pauser, WITHDRAW_DELAY);
        targetContract(address(handler));
    }

    function invariant_solvent() public view {
        assertGe(fly.balanceOf(address(vault)), vault.totalLocked() + vault.totalPending());
    }

    function invariant_exactAccounting() public view {
        assertEq(fly.balanceOf(address(vault)), vault.totalLocked() + vault.totalPending() + handler.donated());
    }

    /// Nobody ever holds (locked + pending) other than exactly what they put in minus what they took out, and nobody
    /// takes out more than they put in.
    function invariant_perActorAccountingIsExact() public view {
        for (uint256 i; i < handler.actorCount(); ++i) {
            address a = handler.actors(i);
            assertLe(handler.withdrawnBy(a), handler.lockedBy(a));
            assertEq(vault.locked(a) + vault.pendingOf(a), handler.lockedBy(a) - handler.withdrawnBy(a));
        }
    }

    function invariant_sumOfLockedIsTotalLocked() public view {
        uint256 sum;
        for (uint256 i; i < handler.actorCount(); ++i) {
            sum += vault.locked(handler.actors(i));
        }
        assertEq(sum, vault.totalLocked());
    }

    function invariant_sumOfPendingIsTotalPending() public view {
        uint256 sum;
        for (uint256 i; i < handler.actorCount(); ++i) {
            address a = handler.actors(i);
            uint256 perUser = vault.pendingOf(a);
            sum += perUser;
            // pendingOf equals the sum of the user's pending requests
            uint256[] memory ids = vault.requestsOf(a);
            uint256 fromRequests;
            for (uint256 j; j < ids.length; ++j) {
                (address user, uint256 amount,, uint8 state) = vault.request(ids[j]);
                assertEq(user, a);
                if (state == 1) fromRequests += amount;
            }
            assertEq(fromRequests, perUser);
        }
        assertEq(sum, vault.totalPending());
    }
}
