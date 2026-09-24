// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {Initializable} from "@openzeppelin/contracts/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts/proxy/utils/UUPSUpgradeable.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import {SafeCast} from "@openzeppelin/contracts/utils/math/SafeCast.sol";
import {AccessControlUpgradeable} from "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import {PausableUpgradeable} from "@openzeppelin/contracts-upgradeable/utils/PausableUpgradeable.sol";

/// @title FlyVault
/// @notice Holders lock $FLY here to share the trading fly's weekly profit, which is paid off-chain and weighted by
///         each holder's locked (earning) balance over time, as reconstructed from this contract's events.
///         Leaving takes two steps: `requestWithdrawal` stops the tokens earning at once and makes them withdrawable
///         `WITHDRAW_DELAY` later; `cancelRequest` puts them back to work instead.
/// @dev UUPS implementation behind an ERC1967Proxy. All state lives in ERC-7201 namespaced storage. Upgrades and role
///      changes go through an OZ TimelockController. The pauser can stop only the calls that start or resume earning
///      (`lock`, `cancelRequest`); leaving (`requestWithdrawal`, `withdraw`) can never be paused.
///      Positions are not transferable and there is no receipt token and no way to sweep the token out.
///
///      ReentrancyGuard: OZ >= 5.5 no longer ships `ReentrancyGuardUpgradeable`; the storage-based (non-transient)
///      `ReentrancyGuard` is proxy-safe (ERC-7201 slot "openzeppelin.storage.ReentrancyGuard", the same slot the old
///      upgradeable variant used; an unset slot reads as "not entered").
contract FlyVault is Initializable, AccessControlUpgradeable, PausableUpgradeable, ReentrancyGuard, UUPSUpgradeable {
    using SafeERC20 for IERC20;

    enum RequestState {
        None,
        Pending,
        Cancelled,
        Withdrawn
    }

    struct Request {
        address user; // slot 0 (20 bytes)
        uint64 readyAt; // slot 0 (8 bytes)
        RequestState state; // slot 0 (1 byte)
        uint256 amount; // slot 1
    }

    /// @custom:storage-location erc7201:fly.storage.FlyVault
    struct FlyVaultStorage {
        IERC20 token;
        uint256 totalLocked;
        uint256 totalPending;
        uint256 lastRequestId;
        mapping(address user => uint256) locked;
        mapping(address user => uint256) pending;
        mapping(uint256 id => Request) requests;
        mapping(address user => uint256[]) requestIds;
    }

    // keccak256(abi.encode(uint256(keccak256("fly.storage.FlyVault")) - 1)) & ~bytes32(uint256(0xff))
    bytes32 private constant FLY_VAULT_STORAGE_LOCATION =
        0x628d6d9a1c3d259b485ba5d3ba03f0dd4cec7e8bba533361079a7332ee4fdb00;

    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");

    /// @notice Seconds between a withdrawal request and the earliest `withdraw`. Immutable in the implementation, so
    ///         changing it takes an upgrade. Requests keep the `readyAt` they were created with.
    /// @custom:oz-upgrades-unsafe-allow state-variable-immutable
    uint256 public immutable WITHDRAW_DELAY;

    event Locked(address indexed user, uint256 amount, uint256 lockedAfter);
    event WithdrawRequested(
        address indexed user, uint256 indexed id, uint256 amount, uint64 readyAt, uint256 lockedAfter
    );
    event RequestCancelled(address indexed user, uint256 indexed id, uint256 amount, uint256 lockedAfter);
    event Withdrawn(address indexed user, uint256 indexed id, uint256 amount);

    error ZeroAddress();
    error ZeroAmount();
    /// @dev The token delivered a different amount than requested (fee-on-transfer or similar); nothing is credited.
    error TransferAmountMismatch(uint256 expected, uint256 received);
    error InsufficientLocked(uint256 requested, uint256 available);
    error UnknownRequest(uint256 id);
    error NotRequestOwner(uint256 id);
    error RequestNotPending(uint256 id);
    error WithdrawalNotReady(uint256 id, uint64 readyAt);

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor(uint256 withdrawDelay) {
        WITHDRAW_DELAY = withdrawDelay;
        _disableInitializers();
    }

    /// @notice One-time setup, run by the proxy's constructor so it cannot be front-run.
    /// @param fly The $FLY token; fixed for the life of the vault.
    /// @param pauser Gets PAUSER_ROLE.
    /// @param timelock Gets DEFAULT_ADMIN_ROLE and UPGRADER_ROLE.
    function initialize(address fly, address pauser, address timelock) external initializer {
        if (fly == address(0) || pauser == address(0) || timelock == address(0)) revert ZeroAddress();
        __AccessControl_init();
        __Pausable_init();
        _getFlyVaultStorage().token = IERC20(fly);
        _grantRole(DEFAULT_ADMIN_ROLE, timelock);
        _grantRole(UPGRADER_ROLE, timelock);
        _grantRole(PAUSER_ROLE, pauser);
    }

    // ------------------------------------------------------------------ holder actions

    /// @notice Lock `amount` $FLY (needs prior approval). Locked tokens earn from this block on.
    function lock(uint256 amount) external whenNotPaused nonReentrant {
        if (amount == 0) revert ZeroAmount();
        FlyVaultStorage storage $ = _getFlyVaultStorage();
        address user = _msgSender();
        IERC20 fly = $.token;

        // Credit only what actually arrived; a token that delivers less (or more) than `amount` is rejected.
        uint256 balanceBefore = fly.balanceOf(address(this));
        fly.safeTransferFrom(user, address(this), amount);
        uint256 received = fly.balanceOf(address(this)) - balanceBefore;
        if (received != amount) revert TransferAmountMismatch(amount, received);

        uint256 lockedAfter = $.locked[user] + amount;
        $.locked[user] = lockedAfter;
        $.totalLocked += amount;
        emit Locked(user, amount, lockedAfter);
    }

    /// @notice Stop `amount` of your locked $FLY earning and make it withdrawable after `WITHDRAW_DELAY`.
    ///         Works while paused.
    /// @return id The new request's global id (ids start at 1).
    function requestWithdrawal(uint256 amount) external returns (uint256 id) {
        if (amount == 0) revert ZeroAmount();
        FlyVaultStorage storage $ = _getFlyVaultStorage();
        address user = _msgSender();
        uint256 available = $.locked[user];
        if (amount > available) revert InsufficientLocked(amount, available);

        uint256 lockedAfter = available - amount;
        $.locked[user] = lockedAfter;
        $.totalLocked -= amount;
        $.pending[user] += amount;
        $.totalPending += amount;

        uint64 readyAt = SafeCast.toUint64(block.timestamp + WITHDRAW_DELAY);
        id = ++$.lastRequestId;
        $.requests[id] = Request({user: user, readyAt: readyAt, state: RequestState.Pending, amount: amount});
        $.requestIds[user].push(id);
        emit WithdrawRequested(user, id, amount, readyAt, lockedAfter);
    }

    /// @notice Cancel your pending request `id`; its tokens are locked (and earning) again. Blocked while paused,
    ///         because it re-locks.
    function cancelRequest(uint256 id) external whenNotPaused {
        FlyVaultStorage storage $ = _getFlyVaultStorage();
        address user = _msgSender();
        Request storage r = _pendingRequestOf($, id, user);
        uint256 amount = r.amount;
        r.state = RequestState.Cancelled;

        $.pending[user] -= amount;
        $.totalPending -= amount;
        uint256 lockedAfter = $.locked[user] + amount;
        $.locked[user] = lockedAfter;
        $.totalLocked += amount;
        emit RequestCancelled(user, id, amount, lockedAfter);
    }

    /// @notice Withdraw your pending request `id` once `readyAt` has been reached. Works while paused.
    function withdraw(uint256 id) external nonReentrant {
        FlyVaultStorage storage $ = _getFlyVaultStorage();
        address user = _msgSender();
        Request storage r = _pendingRequestOf($, id, user);
        uint64 readyAt = r.readyAt;
        if (block.timestamp < readyAt) revert WithdrawalNotReady(id, readyAt);
        uint256 amount = r.amount;
        r.state = RequestState.Withdrawn;

        $.pending[user] -= amount;
        $.totalPending -= amount;
        emit Withdrawn(user, id, amount);
        $.token.safeTransfer(user, amount);
    }

    // ------------------------------------------------------------------ admin

    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }

    // ------------------------------------------------------------------ views

    function token() external view returns (address) {
        return address(_getFlyVaultStorage().token);
    }

    /// @notice The user's earning balance (excludes pending withdrawals).
    function locked(address user) external view returns (uint256) {
        return _getFlyVaultStorage().locked[user];
    }

    function pendingOf(address user) external view returns (uint256) {
        return _getFlyVaultStorage().pending[user];
    }

    function totalLocked() external view returns (uint256) {
        return _getFlyVaultStorage().totalLocked;
    }

    function totalPending() external view returns (uint256) {
        return _getFlyVaultStorage().totalPending;
    }

    /// @return user Owner of the request (zero if `id` is unknown).
    /// @return amount $FLY amount.
    /// @return readyAt Earliest `withdraw` timestamp.
    /// @return state 0 none, 1 pending, 2 cancelled, 3 withdrawn.
    function request(uint256 id) external view returns (address user, uint256 amount, uint64 readyAt, uint8 state) {
        Request storage r = _getFlyVaultStorage().requests[id];
        return (r.user, r.amount, r.readyAt, uint8(r.state));
    }

    /// @notice Every request id the user ever made, oldest first (meant for off-chain reads).
    function requestsOf(address user) external view returns (uint256[] memory ids) {
        return _getFlyVaultStorage().requestIds[user];
    }

    // ------------------------------------------------------------------ internals

    function _pendingRequestOf(FlyVaultStorage storage $, uint256 id, address user)
        private
        view
        returns (Request storage r)
    {
        r = $.requests[id];
        if (r.state == RequestState.None) revert UnknownRequest(id);
        if (r.user != user) revert NotRequestOwner(id);
        if (r.state != RequestState.Pending) revert RequestNotPending(id);
    }

    function _authorizeUpgrade(address) internal override onlyRole(UPGRADER_ROLE) {}

    function _getFlyVaultStorage() private pure returns (FlyVaultStorage storage $) {
        assembly ("memory-safe") {
            $.slot := FLY_VAULT_STORAGE_LOCATION
        }
    }
}
