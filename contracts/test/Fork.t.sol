// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, Vm, console} from "forge-std/Test.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {UUPSUpgradeable} from "@openzeppelin/contracts/proxy/utils/UUPSUpgradeable.sol";
import {FlyVault} from "../src/FlyVault.sol";
import {FlyVaultStack} from "../script/FlyVaultStack.sol";
import {FlyVaultV2} from "./mocks/FlyVaultV2.sol";

/// @notice Runs only with FORK=1: forks Robinhood Chain mainnet and uses the real $FLY.
///   FORK=1 forge test --match-contract ForkTest -vv       (RH_RPC_URL overrides the public RPC)
contract ForkTest is Test {
    /// Uniswap v4 PoolManager on Robinhood Chain; holds the $FLY pool liquidity (~216M FLY on 2026-09-23).
    address constant POOL_MANAGER = 0x8366a39CC670B4001A1121B8F6A443A643e40951;

    IERC20 internal fly = IERC20(FlyVaultStack.FLY_MAINNET);
    address internal owner = makeAddr("owner");
    address internal pauser = makeAddr("pauser");
    address internal alice = makeAddr("alice");
    address internal eveFork = makeAddr("eve");
    FlyVaultStack.Deployment internal d;
    FlyVaultStack.Config internal cfg;
    bool internal enabled;

    function setUp() public {
        string memory flag = vm.envOr("FORK", string(""));
        enabled = keccak256(bytes(flag)) == keccak256("1") || keccak256(bytes(flag)) == keccak256("true");
        if (!enabled) return;
        vm.createSelectFork(vm.envOr("RH_RPC_URL", string("https://rpc.mainnet.chain.robinhood.com")));
        assertEq(block.chainid, FlyVaultStack.ROBINHOOD_CHAIN_ID);
        cfg = FlyVaultStack.Config({
            fly: address(fly),
            owner: owner,
            pauser: pauser,
            timelockDelay: FlyVaultStack.MAINNET_TIMELOCK_DELAY,
            withdrawDelay: FlyVaultStack.MAINNET_WITHDRAW_DELAY
        });
        FlyVaultStack.enforceChainRules(cfg); // what Deploy.s.sol runs on chain 4663
        d = FlyVaultStack.deploy(cfg);
        FlyVaultStack.check(d, cfg, address(this));
    }

    modifier onlyFork() {
        if (!enabled) vm.skip(true, "set FORK=1 to run against Robinhood Chain mainnet");
        _;
    }

    function _fundFromRealHolder(address to, uint256 amount) internal {
        uint256 pmBalance = fly.balanceOf(POOL_MANAGER);
        console.log("PoolManager FLY balance", pmBalance);
        if (pmBalance >= amount) {
            vm.prank(POOL_MANAGER);
            assertTrue(fly.transfer(to, amount));
        } else {
            deal(address(fly), to, amount);
        }
        assertEq(fly.balanceOf(to), amount);
    }

    function test_fork_tokenIsWhatWeExpect() public onlyFork {
        assertEq(IERC20Meta(address(fly)).decimals(), 18);
        assertEq(fly.totalSupply(), 1_000_000_000 ether);
        assertEq(d.vault.token(), address(fly));
    }

    function test_fork_lockRequestWithdrawWithRealFly() public onlyFork {
        uint256 amount = 1_000_000 ether;
        _fundFromRealHolder(alice, amount);
        FlyVault vault = d.vault;

        vm.startPrank(alice);
        fly.approve(address(vault), amount);
        vault.lock(amount); // the real token passes the received == amount check
        assertEq(vault.locked(alice), amount);
        assertEq(fly.balanceOf(address(vault)), amount);

        uint256 id = vault.requestWithdrawal(400_000 ether);
        (,, uint64 readyAt,) = vault.request(id);
        assertEq(readyAt, block.timestamp + 7 days);
        vm.warp(readyAt - 1);
        vm.expectRevert(abi.encodeWithSelector(FlyVault.WithdrawalNotReady.selector, id, readyAt));
        vault.withdraw(id);
        vm.warp(readyAt);
        vault.withdraw(id);
        vm.stopPrank();

        assertEq(fly.balanceOf(alice), 400_000 ether);
        assertEq(fly.balanceOf(address(vault)), 600_000 ether);
        assertEq(vault.totalLocked(), 600_000 ether);
        assertEq(vault.totalPending(), 0);
    }

    function test_fork_upgradeThroughTimelock() public onlyFork {
        _fundFromRealHolder(alice, 10 ether);
        vm.startPrank(alice);
        fly.approve(address(d.vault), 10 ether);
        d.vault.lock(10 ether);
        vm.stopPrank();

        FlyVaultV2 v2 = new FlyVaultV2(7 days);
        bytes memory data = abi.encodeCall(UUPSUpgradeable.upgradeToAndCall, (address(v2), ""));
        vm.prank(owner);
        d.timelock.schedule(address(d.vault), 0, data, bytes32(0), bytes32(0), 8 days);
        skip(8 days);
        d.timelock.execute(address(d.vault), 0, data, bytes32(0), bytes32(0));
        assertEq(FlyVaultV2(address(d.vault)).version(), 2);
        assertEq(d.vault.locked(alice), 10 ether);
    }

    /// The real $FLY has no hooks, owner, mint, pause or blacklist that could touch the vault: its bytecode is not a
    /// proxy, a transfer into and out of the vault touches only the token's own storage, and nobody can burn the
    /// vault's balance (burnFrom needs an allowance the vault never gives).
    function test_fork_realFlyCannotReachIntoTheVault() public onlyFork {
        bytes32 implSlot = 0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc;
        assertEq(vm.load(address(fly), implSlot), bytes32(0), "FLY is an ERC-1967 proxy");
        _fundFromRealHolder(alice, 1000 ether);
        FlyVault vault = d.vault;

        vm.startPrank(alice);
        fly.approve(address(vault), 1000 ether);
        vm.startStateDiffRecording();
        vault.lock(1000 ether);
        uint256 id = vault.requestWithdrawal(1000 ether);
        vm.warp(block.timestamp + 7 days);
        vault.withdraw(id);
        Vm.AccountAccess[] memory acc = vm.stopAndReturnStateDiff();
        vm.stopPrank();
        for (uint256 i; i < acc.length; ++i) {
            address a = acc[i].account;
            if (a == address(vm)) continue; // the vm.warp cheatcode call itself
            assertTrue(
                a == address(vault) || a == address(fly) || a == address(d.implementation),
                "unexpected contract touched during lock/withdraw"
            );
        }
        assertEq(fly.balanceOf(alice), 1000 ether);

        address carol = makeAddr("carol");
        _fundFromRealHolder(carol, 5 ether);
        vm.startPrank(carol);
        fly.approve(address(vault), 5 ether);
        vault.lock(5 ether);
        vm.stopPrank();
        vm.prank(eveFork);
        (bool ok,) = address(fly).call(abi.encodeWithSignature("burnFrom(address,uint256)", address(vault), 1));
        assertFalse(ok, "burnFrom on the vault succeeded");
        assertEq(fly.allowance(address(vault), eveFork), 0);
        assertEq(fly.balanceOf(address(vault)), 5 ether);
    }

    function test_fork_mainnetRulesRefuseOtherParams() public onlyFork {
        FlyVaultStack.Config memory bad = cfg;
        bad.withdrawDelay = 1 days;
        vm.expectRevert(bytes("FlyVaultStack: mainnet WITHDRAW_DELAY must be 7 days"));
        this.enforce(bad);
        bad = cfg;
        bad.timelockDelay = 2 days;
        vm.expectRevert(bytes("FlyVaultStack: mainnet TIMELOCK_DELAY must be 8 days"));
        this.enforce(bad);
        bad = cfg;
        bad.fly = address(0xBEEF);
        vm.expectRevert(bytes("FlyVaultStack: mainnet FLY_TOKEN must be the real $FLY"));
        this.enforce(bad);
    }

    function enforce(FlyVaultStack.Config memory c) external view {
        FlyVaultStack.enforceChainRules(c);
    }
}

interface IERC20Meta {
    function decimals() external view returns (uint8);
}
