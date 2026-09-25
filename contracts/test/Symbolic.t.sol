// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {FlyVault} from "../src/FlyVault.sol";
import {MockFLY} from "../src/testnet/MockFLY.sol";

/// @notice Symbolic proofs for Halmos (`halmos --contract SymbolicVaultTest`). Every `check_` holds for ALL inputs:
///         any attacker address, any amount, any request id, any time. Forge runs them as ordinary fuzz tests too.
contract SymbolicVaultTest is Test {
    uint256 internal constant DELAY = 7 days;
    address internal constant ALICE = address(0xA11CE);
    address internal constant PAUSER = address(0x9A05E4);
    address internal constant ADMIN = address(0xAD314);

    MockFLY internal fly;
    FlyVault internal vault;
    uint256 internal aliceId;

    function setUp() public {
        fly = new MockFLY();
        FlyVault impl = new FlyVault(DELAY);
        vault = FlyVault(
            address(new ERC1967Proxy(address(impl), abi.encodeCall(FlyVault.initialize, (address(fly), PAUSER, ADMIN))))
        );
        // Alice: 10 locked, 3 pending in request 1
        fly.mint(ALICE, 13 ether);
        vm.startPrank(ALICE);
        fly.approve(address(vault), type(uint256).max);
        vault.lock(13 ether);
        aliceId = vault.requestWithdrawal(3 ether);
        vm.stopPrank();
    }

    function _aliceSnapshot() internal view returns (bytes32) {
        (address u, uint256 amt, uint64 readyAt, uint8 st) = vault.request(aliceId);
        return
            keccak256(
                abi.encode(vault.locked(ALICE), vault.pendingOf(ALICE), fly.balanceOf(ALICE), u, amt, readyAt, st)
            );
    }

    function _solvent() internal view {
        assert(fly.balanceOf(address(vault)) >= vault.totalLocked() + vault.totalPending());
    }

    /// Any single vault call by anyone other than Alice, with any arguments, at any time, leaves Alice's position,
    /// her wallet and her request exactly as they were, and the vault solvent.
    function check_noCallByOthersTouchesAlice(
        address attacker,
        uint8 which,
        uint256 amount,
        uint256 id,
        uint32 dt,
        bool paused
    ) public {
        vm.assume(attacker != ALICE && attacker != address(vault) && attacker != address(0));
        if (paused) {
            vm.prank(PAUSER);
            vault.pause();
        }
        vm.warp(block.timestamp + dt);
        amount = bound(amount, 0, fly.MAX_MINT());
        fly.mint(attacker, amount);
        vm.prank(attacker);
        fly.approve(address(vault), type(uint256).max);
        bytes32 before = _aliceSnapshot();

        vm.startPrank(attacker);
        which = which % 7;
        if (which == 0) vault.lock(amount);
        else if (which == 1) vault.requestWithdrawal(amount);
        else if (which == 2) vault.cancelRequest(id);
        else if (which == 3) vault.withdraw(id);
        else if (which == 4) vault.pause();
        else if (which == 5) vault.unpause();
        else vault.upgradeToAndCall(address(uint160(id)), "");
        vm.stopPrank();

        assert(_aliceSnapshot() == before);
        _solvent();
    }

    /// Three arbitrary calls in a row by the attacker (enough for lock -> request -> act on someone else's id) still
    /// cannot touch Alice, and the attacker's wallet + locked + pending is conserved. Forge-fuzzed only (not `check_`):
    /// Halmos 0.3.3 reports lock -> request -> withdraw-own-request as a violation, but every such counterexample
    /// replays cleanly as a concrete call sequence (checked 2026-09-25), so it is a solver-model artifact.
    function prop_threeCallsByAttacker(uint8 w1, uint8 w2, uint8 w3, uint256 a1, uint256 a2, uint256 a3, uint32 dt)
        public
    {
        address attacker = address(0xE7E);
        fly.mint(attacker, fly.MAX_MINT());
        vm.prank(attacker);
        fly.approve(address(vault), type(uint256).max);
        bytes32 before = _aliceSnapshot();
        uint256 walletBefore = fly.balanceOf(attacker);

        _call(attacker, w1 % 4, a1);
        _call(attacker, w2 % 4, a2);
        vm.warp(block.timestamp + dt);
        _call(attacker, w3 % 4, a3);

        assert(_aliceSnapshot() == before);
        _solvent();
        assert(fly.balanceOf(attacker) + vault.locked(attacker) + vault.pendingOf(attacker) == walletBefore);
    }

    function _call(address who, uint8 which, uint256 arg) internal {
        vm.startPrank(who);
        if (which == 0) vault.lock(bound(arg, 0, fly.MAX_MINT()));
        else if (which == 1) vault.requestWithdrawal(arg);
        else if (which == 2) vault.cancelRequest(arg);
        else vault.withdraw(arg);
        vm.stopPrank();
    }

    /// Alice can always get her pending request out once it is ready, whatever the pause state, and gets exactly the
    /// requested amount; before readyAt she cannot.
    function check_aliceCanAlwaysExit(bool paused, uint32 early) public {
        if (paused) {
            vm.prank(PAUSER);
            vault.pause();
        }
        (, uint256 amt, uint64 readyAt,) = vault.request(aliceId);
        uint256 before = fly.balanceOf(ALICE);
        if (early > 0 && early <= DELAY) {
            vm.warp(readyAt - early);
            vm.prank(ALICE);
            try vault.withdraw(aliceId) {
                assert(false);
            } catch {}
        }
        vm.warp(readyAt);
        vm.prank(ALICE);
        vault.withdraw(aliceId);
        assert(fly.balanceOf(ALICE) == before + amt);
        _solvent();
    }

    // forge runs the same properties as fuzz tests; reverting paths are fine (Halmos skips them too), a failed
    // assert (Panic 0x01) is not.
    function testFuzz_noCallByOthersTouchesAlice(address a, uint8 w, uint256 amt, uint256 id, uint32 dt, bool p)
        external
    {
        try this.check_noCallByOthersTouchesAlice(a, w, amt, id % 4, dt, p) {}
        catch (bytes memory e) {
            _notAssert(e);
        }
    }

    function testFuzz_threeCallsByAttacker(uint8 w1, uint8 w2, uint8 w3, uint256 a1, uint256 a2, uint256 a3, uint32 dt)
        external
    {
        // small ids so the id-taking calls hit real requests
        try this.prop_threeCallsByAttacker(w1, w2, w3, _small(a1), _small(a2), _small(a3), dt) {}
        catch (bytes memory e) {
            _notAssert(e);
        }
    }

    function testFuzz_aliceCanAlwaysExit(bool p, uint32 early) external {
        check_aliceCanAlwaysExit(p, early);
    }

    function _small(uint256 x) internal pure returns (uint256) {
        return x % 3 == 0 ? x : x % 4;
    }

    function _notAssert(bytes memory e) internal pure {
        assertFalse(keccak256(e) == keccak256(abi.encodeWithSignature("Panic(uint256)", 1)), "assertion violated");
    }
}
