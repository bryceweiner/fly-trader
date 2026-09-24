// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {TimelockController} from "@openzeppelin/contracts/governance/TimelockController.sol";
import {HandOver} from "../script/HandOver.s.sol";
import {VaultTest} from "./utils/VaultTest.sol";

contract HandOverTest is VaultTest {
    address internal hw = makeAddr("hardware");

    function _batch() internal returns (address[] memory t, uint256[] memory v, bytes[] memory p) {
        return new HandOver().batch(address(vault), address(timelock), owner, hw);
    }

    function test_handOver_movesEveryRoleAfterTheDelay() public {
        // the temporary key is also the pauser on mainnet (Deploy.s.sol: PAUSER defaults to OWNER)
        vm.startPrank(address(timelock));
        vault.grantRole(vault.PAUSER_ROLE(), owner);
        vm.stopPrank();

        (address[] memory t, uint256[] memory v, bytes[] memory p) = _batch();
        vm.prank(owner);
        timelock.scheduleBatch(t, v, p, bytes32(0), bytes32(0), TIMELOCK_DELAY);

        skip(TIMELOCK_DELAY - 1);
        vm.expectRevert();
        timelock.executeBatch(t, v, p, bytes32(0), bytes32(0));
        skip(1);
        vm.prank(makeAddr("anyone"));
        timelock.executeBatch(t, v, p, bytes32(0), bytes32(0));

        assertTrue(timelock.hasRole(timelock.PROPOSER_ROLE(), hw));
        assertTrue(timelock.hasRole(timelock.CANCELLER_ROLE(), hw));
        assertTrue(vault.hasRole(vault.PAUSER_ROLE(), hw));
        assertFalse(timelock.hasRole(timelock.PROPOSER_ROLE(), owner));
        assertFalse(timelock.hasRole(timelock.CANCELLER_ROLE(), owner));
        assertFalse(vault.hasRole(vault.PAUSER_ROLE(), owner));

        // the old key can no longer propose; the hardware wallet can
        vm.prank(owner);
        vm.expectRevert();
        timelock.schedule(address(vault), 0, "", bytes32(0), bytes32(uint256(1)), TIMELOCK_DELAY);
        vm.prank(hw);
        timelock.schedule(address(vault), 0, "", bytes32(0), bytes32(uint256(1)), TIMELOCK_DELAY);
        vm.prank(hw);
        vault.pause();
    }

    function test_handOver_cannotBeShortcut() public {
        (address[] memory t, uint256[] memory v, bytes[] memory p) = _batch();
        vm.prank(owner);
        vm.expectRevert();                       // below the minimum delay
        timelock.scheduleBatch(t, v, p, bytes32(0), bytes32(0), TIMELOCK_DELAY - 1);
        vm.prank(hw);
        vm.expectRevert();                       // the hardware wallet has no power before the hand-over
        timelock.scheduleBatch(t, v, p, bytes32(0), bytes32(0), TIMELOCK_DELAY);
    }
}
