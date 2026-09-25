// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {FlyVault} from "../../src/FlyVault.sol";
import {MockFLY} from "../../src/testnet/MockFLY.sol";

/// @notice One holder. Only the harness drives it, so `msg.sender` inside the vault is this contract.
contract Actor {
    FlyVault internal immutable vault;
    MockFLY internal immutable fly;

    constructor(FlyVault v, MockFLY f) {
        (vault, fly) = (v, f);
        f.approve(address(v), type(uint256).max);
    }

    function lock(uint256 a) external {
        vault.lock(a);
    }

    function request(uint256 a) external returns (uint256) {
        return vault.requestWithdrawal(a);
    }

    function cancel(uint256 id) external {
        vault.cancelRequest(id);
    }

    function withdraw(uint256 id) external {
        vault.withdraw(id);
    }

    function pause() external {
        vault.pause();
    }

    function unpause() external {
        vault.unpause();
    }

    function upgrade(address impl) external {
        vault.upgradeToAndCall(impl, "");
    }
}

/// @notice Echidna property harness (not run by forge; forge only compiles it).
///         echidna test/echidna/EchidnaVault.sol --contract EchidnaVault --config test/echidna/echidna.yaml
///         Echidna drives every public function below with random arguments, senders and block times.
///         Properties: the vault always holds exactly what it owes; each holder's tokens are conserved; nobody but
///         the owner of a request can move it; a ready request always withdraws; no one without a role can pause
///         or upgrade; paused vaults still let everyone leave.
contract EchidnaVault {
    uint256 internal constant DELAY = 7 days;
    uint256 internal constant START = 1_000_000 ether;
    uint256 internal constant N = 3;

    MockFLY internal fly;
    FlyVault internal vault;
    Actor internal pauserActor;
    Actor[N] internal actors;
    uint256 internal donated;
    uint256 internal reqs; // ids are global and only actors request, so this is the vault's lastRequestId
    bool internal readyWithdrawFailed;
    bool internal outsiderMovedRequest;
    bool internal outsiderGotRole;

    constructor() {
        fly = new MockFLY();
        FlyVault impl = new FlyVault(DELAY);
        // The harness itself is the "timelock" (admin + upgrader); the pauser is a separate actor.
        address pauserAddr = address(uint160(uint256(keccak256("pauser-placeholder"))));
        vault = FlyVault(
            address(
                new ERC1967Proxy(
                    address(impl), abi.encodeCall(FlyVault.initialize, (address(fly), pauserAddr, address(this)))
                )
            )
        );
        pauserActor = new Actor(vault, fly);
        vault.grantRole(vault.PAUSER_ROLE(), address(pauserActor));
        vault.revokeRole(vault.PAUSER_ROLE(), pauserAddr);
        for (uint256 i; i < N; ++i) {
            actors[i] = new Actor(vault, fly);
            fly.mint(address(actors[i]), START);
        }
    }

    function _actor(uint256 i) internal view returns (Actor) {
        return actors[i % N];
    }

    // ------------------------------------------------------------------ actions

    function lock(uint256 i, uint256 amount) external {
        Actor a = _actor(i);
        amount = amount % (fly.balanceOf(address(a)) + 1);
        try a.lock(amount) {} catch {}
    }

    function request(uint256 i, uint256 amount) external {
        Actor a = _actor(i);
        amount = amount % (vault.locked(address(a)) + 1);
        try a.request(amount) returns (uint256 id) {
            reqs = id;
        } catch {}
    }

    function cancel(uint256 i, uint256 id) external {
        Actor a = _actor(i);
        id = id % (reqs + 2);
        (address owner,,, uint8 st) = vault.request(id);
        uint256 lockedBefore = vault.locked(owner);
        try a.cancel(id) {
            if (owner != address(a) || st != 1) outsiderMovedRequest = true;
        } catch {}
        if (owner != address(a) && vault.locked(owner) != lockedBefore) outsiderMovedRequest = true;
    }

    function withdraw(uint256 i, uint256 id) external {
        Actor a = _actor(i);
        id = id % (reqs + 2);
        (address owner, uint256 amount, uint64 readyAt, uint8 st) = vault.request(id);
        uint256 before = fly.balanceOf(address(a));
        try a.withdraw(id) {
            if (owner != address(a) || st != 1 || block.timestamp < readyAt) outsiderMovedRequest = true;
            if (fly.balanceOf(address(a)) != before + amount) readyWithdrawFailed = true;
        } catch {
            // a pending, ready request of your own must always withdraw, paused or not
            if (owner == address(a) && st == 1 && block.timestamp >= readyAt) readyWithdrawFailed = true;
        }
    }

    function togglePause(bool on) external {
        if (on) {
            try pauserActor.pause() {} catch {}
        } else {
            try pauserActor.unpause() {} catch {}
        }
    }

    /// Anyone other than the pauser / timelock tries the privileged calls.
    function outsiderPrivileged(uint256 i, uint8 which) external {
        Actor a = _actor(i);
        which = which % 3;
        if (which == 0) {
            try a.pause() {
                outsiderGotRole = true;
            } catch {}
        } else if (which == 1) {
            try a.unpause() {
                outsiderGotRole = true;
            } catch {}
        } else {
            FlyVault evil = new FlyVault(0);
            try a.upgrade(address(evil)) {
                outsiderGotRole = true;
            } catch {}
        }
    }

    function donate(uint256 amount) external {
        amount = amount % 1_000 ether;
        fly.mint(address(vault), amount);
        donated += amount;
    }

    // ------------------------------------------------------------------ properties

    function echidna_solvent() external view returns (bool) {
        return fly.balanceOf(address(vault)) == vault.totalLocked() + vault.totalPending() + donated;
    }

    function echidna_totalsMatchActors() external view returns (bool) {
        uint256 l;
        uint256 p;
        for (uint256 i; i < N; ++i) {
            l += vault.locked(address(actors[i]));
            p += vault.pendingOf(address(actors[i]));
        }
        return l == vault.totalLocked() && p == vault.totalPending();
    }

    function echidna_eachHolderConserved() external view returns (bool) {
        for (uint256 i; i < N; ++i) {
            address a = address(actors[i]);
            if (fly.balanceOf(a) + vault.locked(a) + vault.pendingOf(a) != START) return false;
        }
        return true;
    }

    function echidna_pendingMatchesRequests() external view returns (bool) {
        uint256[N] memory sums;
        uint256 n = reqs;
        for (uint256 id = 1; id <= n; ++id) {
            (address owner, uint256 amount,, uint8 st) = vault.request(id);
            if (st != 1) continue;
            for (uint256 i; i < N; ++i) {
                if (owner == address(actors[i])) sums[i] += amount;
            }
        }
        for (uint256 i; i < N; ++i) {
            if (sums[i] != vault.pendingOf(address(actors[i]))) return false;
        }
        return true;
    }

    function echidna_readyWithdrawalsAlwaysPay() external view returns (bool) {
        return !readyWithdrawFailed;
    }

    function echidna_noOneMovesAnotherHoldersRequest() external view returns (bool) {
        return !outsiderMovedRequest;
    }

    function echidna_rolesStayPut() external view returns (bool) {
        return !outsiderGotRole;
    }
}
