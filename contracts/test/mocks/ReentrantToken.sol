// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";

/// @notice Token with a transfer hook that calls back into a target (the vault) mid-transfer, like an ERC-777 hook.
contract ReentrantToken is ERC20 {
    address public target;
    bytes public callbackData;

    constructor() ERC20("Reentrant Token", "REE") {}

    function mint(address to, uint256 amount) external {
        _mint(to, amount);
    }

    function arm(address target_, bytes calldata data) external {
        target = target_;
        callbackData = data;
    }

    function _update(address from, address to, uint256 value) internal override {
        super._update(from, to, value);
        address t = target;
        if (t != address(0) && from != address(0)) {
            target = address(0); // fire once
            (bool ok, bytes memory ret) = t.call(callbackData);
            if (!ok) {
                assembly ("memory-safe") {
                    revert(add(ret, 32), mload(ret))
                }
            }
        }
    }
}
