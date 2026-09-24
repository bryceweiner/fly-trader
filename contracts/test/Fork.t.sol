// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console} from "forge-std/Test.sol";
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
