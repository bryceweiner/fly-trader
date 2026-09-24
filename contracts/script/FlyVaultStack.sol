// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {TimelockController} from "@openzeppelin/contracts/governance/TimelockController.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import {FlyVault} from "../src/FlyVault.sol";

/// @notice The one code path that builds the vault stack: used by the deploy scripts and by the tests.
library FlyVaultStack {
    uint256 internal constant ROBINHOOD_CHAIN_ID = 4663;
    uint256 internal constant ROBINHOOD_TESTNET_CHAIN_ID = 46630;
    uint256 internal constant ANVIL_CHAIN_ID = 31337;
    address internal constant FLY_MAINNET = 0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3;
    uint256 internal constant MAINNET_TIMELOCK_DELAY = 8 days;
    uint256 internal constant MAINNET_WITHDRAW_DELAY = 7 days;

    struct Config {
        address fly;
        address owner; // timelock proposer + canceller (hardware wallet)
        address pauser; // vault PAUSER_ROLE
        uint256 timelockDelay;
        uint256 withdrawDelay;
    }

    struct Deployment {
        TimelockController timelock;
        FlyVault implementation;
        FlyVault vault; // the proxy: the address everyone uses
    }

    /// @notice Refuses anything but the frozen mainnet parameters on Robinhood Chain.
    function enforceChainRules(Config memory c) internal view {
        require(c.fly != address(0) && c.owner != address(0) && c.pauser != address(0), "FlyVaultStack: zero address");
        if (block.chainid == ROBINHOOD_CHAIN_ID) {
            require(c.fly == FLY_MAINNET, "FlyVaultStack: mainnet FLY_TOKEN must be the real $FLY");
            require(c.timelockDelay == MAINNET_TIMELOCK_DELAY, "FlyVaultStack: mainnet TIMELOCK_DELAY must be 8 days");
            require(c.withdrawDelay == MAINNET_WITHDRAW_DELAY, "FlyVaultStack: mainnet WITHDRAW_DELAY must be 7 days");
        }
    }

    /// @notice Timelock (proposer = canceller = owner, executor = anyone, no admin), implementation, and proxy
    ///         initialized in its own constructor, so initialization cannot be front-run.
    function deploy(Config memory c) internal returns (Deployment memory d) {
        address[] memory proposers = new address[](1);
        proposers[0] = c.owner;
        address[] memory executors = new address[](1);
        executors[0] = address(0); // open role: anyone may execute a ready operation
        d.timelock = new TimelockController(c.timelockDelay, proposers, executors, address(0));
        d.implementation = new FlyVault(c.withdrawDelay);
        bytes memory init = abi.encodeCall(FlyVault.initialize, (c.fly, c.pauser, address(d.timelock)));
        d.vault = FlyVault(address(new ERC1967Proxy(address(d.implementation), init)));
    }

    /// @notice Post-deploy assertions; `deployer` must end up with no power over either contract.
    function check(Deployment memory d, Config memory c, address deployer) internal view {
        FlyVault v = d.vault;
        TimelockController t = d.timelock;
        require(v.token() == c.fly, "check: token");
        require(v.WITHDRAW_DELAY() == c.withdrawDelay, "check: withdraw delay");
        require(v.hasRole(v.DEFAULT_ADMIN_ROLE(), address(t)), "check: vault admin");
        require(v.hasRole(v.UPGRADER_ROLE(), address(t)), "check: vault upgrader");
        require(v.hasRole(v.PAUSER_ROLE(), c.pauser), "check: vault pauser");
        require(!v.paused(), "check: paused");
        require(t.getMinDelay() == c.timelockDelay, "check: timelock delay");
        require(t.hasRole(t.PROPOSER_ROLE(), c.owner), "check: proposer");
        require(t.hasRole(t.CANCELLER_ROLE(), c.owner), "check: canceller");
        require(t.hasRole(t.EXECUTOR_ROLE(), address(0)), "check: open executor");
        require(t.hasRole(t.DEFAULT_ADMIN_ROLE(), address(t)), "check: timelock self-admin");
        require(!t.hasRole(t.DEFAULT_ADMIN_ROLE(), c.owner), "check: owner is timelock admin");
        require(!v.hasRole(v.DEFAULT_ADMIN_ROLE(), c.owner), "check: owner is vault admin");
        require(!v.hasRole(v.UPGRADER_ROLE(), c.owner), "check: owner is vault upgrader");
        if (deployer != address(t)) {
            require(!t.hasRole(t.DEFAULT_ADMIN_ROLE(), deployer), "check: deployer is timelock admin");
            require(!v.hasRole(v.DEFAULT_ADMIN_ROLE(), deployer), "check: deployer is vault admin");
            require(!v.hasRole(v.UPGRADER_ROLE(), deployer), "check: deployer is vault upgrader");
        }
    }
}
