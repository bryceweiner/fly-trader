"""A plain Uniswap v4 memecoin's contract, checked before its pools are indexed (rh/v4_universe.py): what its owner can
do to holders.

Measured 2026-10-23 on the 118 tokens of Robinhood Chain pools that cleared the floors ($97M of 24 h volume): 76 % of the
volume had no owner or admin powers (every Pons coin, Clanker, Pons V1, Pools.trade); the rest held upgradeable sell gates
and blacklists under Pons tickers, owner-set taxes, bridge peers that mint, and built-in taxes.

``check(rpc, token)`` reads the code (a minimal proxy is followed to its implementation; an EIP-1967 or beacon proxy is
upgradeable and refused) and the dispatcher's selectors, and refuses a token whose owner (``owner()``, not renounced) or
AccessControl roles hold a power (``POWERS``), one with a built-in buy or sell tax (``TAXES``), one that is not 18 decimals,
and one whose name or symbol marks a major, a stablecoin or a tokenized stock. Functions are matched by selector: a power
under an unusual name is missed, which the floors and the creator history (rh/meta.py) still stand behind.
"""
from __future__ import annotations

import re

from . import abi

POWERS: dict[str, tuple[str, ...]] = {
    "mint": ("mint(address,uint256)", "mint(uint256)", "mintTo(address,uint256)", "grantMintRole(address)", "grantMintAndBurnRoles(address)",
             "setMinter(address,bool)", "addMinter(address)", "setPeer(uint32,bytes32)"),
    "upgrade": ("upgradeTo(address)", "upgradeToAndCall(address,bytes)", "upgradeToSellGate()", "upgradeToBlacklist()"),
    "blacklist": ("setBlacklist(address,bool)", "setBlacklistBatch(address[],bool)", "addBlacklistAddress(address)", "blacklist(address)",
                  "addToBlacklist(address)", "blacklistAddress(address,bool)", "setSellTarget(address,bool)", "setSellTargets(address[],bool)",
                  "blockVenue(address)", "setTransferAllowed(address,bool)", "setBot(address,bool)", "setBots(address[],bool)", "addBots(address[])",
                  "freeze(address)"),
    "pause": ("pause()",),
    "tax setter": ("setFee(uint256)", "setFees(uint256,uint256)", "setTaxes(uint256,uint256)", "setTaxBps(uint16,uint16)", "setProjectTaxRates(uint16,uint16)",
                   "setRateControl(uint16,uint16)", "setBuyFee(uint256)", "setSellFee(uint256)", "updateFees(uint256,uint256)"),
    "transfer hook": ("setHook(address)", "setGateHook(address,bool)", "setPoolManagerGate(bool,bool)", "setMsgInspector(address)"),
    "trading switch": ("enableTrading()", "openTrading()", "setMaxTxAmount(uint256)", "setMaxWalletSize(uint256)", "setLaunchLimits(uint16,uint16)",
                       "lockPool(address)", "unlockPool()", "setLimitExempt(address,bool)", "setWalletExempt(address,bool)", "disableBalanceLimit()",
                       "enableTransfers()"),
}
TAXES = ("buyTax()", "sellTax()", "buyFee()", "sellFee()", "taxWallet()", "buyTaxBps()", "sellTaxBps()", "totalBuyFeeBasisPoints()", "totalSellFeeBasisPoints()",
         "projectBuyTaxBasisPoints()", "projectSellTaxBasisPoints()", "totalBuyTaxBasisPoints()", "totalSellTaxBasisPoints()", "buyFeePercent()",
         "sellFeePercent()", "buyFeeBasisPoints()", "sellFeeBasisPoints()", "taxFee()", "getTaxConfig()")
ROLES = ("hasRole(bytes32,address)", "grantRole(bytes32,address)")
SEL: dict[str, tuple[str, str]] = {abi.selector(s).hex(): (cat, s) for cat, sigs in POWERS.items() for s in sigs}
TAX_SEL = {abi.selector(s).hex() for s in TAXES}
ROLE_SEL = {abi.selector(s).hex() for s in ROLES}
OWNER_SEL = abi.selector("owner()").hex()
IMPL_SLOT = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"       # EIP-1967 implementation
BEACON_SLOT = "0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50"     # EIP-1967 beacon
CLONE_PREFIXES = ("363d3d373d3d3d363d73", "365f5f375f5f365f73", "3d3d3d3d363d3d37363d73")  # EIP-1167, its PUSH0 form, 0age's
DEAD = {0, 0xdead, int("000000000000000000000000000000000000dEaD", 16)}
MAJORS = {"ETH", "WETH", "USDG", "USDC", "USDT", "U", "USD", "DAI", "PYUSD", "USDE", "CBBTC", "WBTC", "HBTC", "BTC", "TBTC", "WSTETH", "STETH", "WEETH",
          "EZETH", "RETH", "CBETH", "LINK", "HYPE", "TAO", "SOL", "WSOL", "ARB", "OP", "UNI", "AAVE", "VIRTUAL", "LIT", "RAM", "UP", "PONS"}
STOCK_NAME = re.compile(r"robinhood token|tokeni[sz]ed|xstock|ondo|wrapped|bridged|\bstock\b", re.I)


def selectors(code_hex: str) -> set[str]:
    """The dispatcher's function selectors: each PUSH4 followed by EQ, DUP2, GT or LT, walking opcodes (PUSH data skipped)."""
    b = bytes.fromhex(code_hex[2:] if code_hex.startswith("0x") else code_hex); i = 0; out = set()
    while i < len(b):
        op = b[i]
        if op == 0x63 and i + 5 < len(b) and b[i + 5] in (0x14, 0x81, 0x11, 0x10):
            out.add(b[i + 1:i + 5].hex())
        i += 1 + (op - 0x5f if 0x60 <= op <= 0x7f else 0)
    return out


def clone_target(code_hex: str) -> str | None:
    """The implementation a minimal proxy delegates to (EIP-1167 and its two shorter forms), else None."""
    c = code_hex[2:] if code_hex.startswith("0x") else code_hex
    for p in CLONE_PREFIXES:
        if c.startswith(p) and len(c) >= len(p) + 40:
            return "0x" + c[len(p):len(p) + 40]
    return None


def judge(sels: set[str], *, owner_live: bool, upgradeable: bool, decimals: int | None, symbol: str | None, name: str | None) -> tuple[bool, str, list[str]]:
    """(accepted, reason, the powers found). Pure: ``check`` reads what it needs from the chain."""
    powers = sorted({SEL[s][0] for s in sels if s in SEL})
    if upgradeable:
        return False, "upgradeable proxy: its owner can replace the code", powers
    if decimals != 18:
        return False, f"{decimals} decimals (only 18 are traded)", powers
    sym = (symbol or "").strip().upper(); nm = name or ""
    if sym in MAJORS or STOCK_NAME.search(nm):
        return False, f"a major, stablecoin or tokenized asset ({symbol}, {name})", powers
    if sels & TAX_SEL:
        return False, "a built-in buy or sell tax", powers
    if powers and (owner_live or sels & ROLE_SEL):
        return False, ("its owner" if owner_live else "an admin role") + " can " + ", ".join(powers), powers
    return True, "no owner or admin powers" if not powers else "powers with a renounced owner (" + ", ".join(powers) + ")", powers


def _word_address(v) -> int:
    try:
        return int(v, 16) if isinstance(v, str) and v.startswith("0x") else 0
    except ValueError:
        return 0


def _str(v) -> str | None:
    if not isinstance(v, str) or len(v) < 2 + 128:
        return None
    try:
        return abi.decode(["string"], bytes.fromhex(v[2:]))[0]
    except Exception:
        return None


def check(rpc, token: str) -> dict:
    """The contract check of ``token`` (two batched RPC reads at most). ``ok`` False with ``reason`` when refused; an
    unreadable token raises (the caller retries later)."""
    t = token.lower()
    call = lambda sig: ("eth_call", [{"to": t, "data": "0x" + abi.selector(sig).hex()}, "latest"])      # noqa: E731
    r = rpc.batch([("eth_getCode", [t, "latest"]), ("eth_getStorageAt", [t, IMPL_SLOT, "latest"]), ("eth_getStorageAt", [t, BEACON_SLOT, "latest"]),
                   call("decimals()"), call("symbol()"), call("name()"), call("totalSupply()"), call("owner()")])
    code = r[0] if isinstance(r[0], str) else None
    if not code or code == "0x":
        raise RuntimeError(f"no code at {t}")
    ok = lambda v: isinstance(v, str)                                                                   # noqa: E731  (an EvmRpcError otherwise)
    upgradeable = (ok(r[1]) and _word_address(r[1]) != 0) or (ok(r[2]) and _word_address(r[2]) != 0)
    impl = clone_target(code); body = code
    if impl:
        body = rpc.batch([("eth_getCode", [impl, "latest"])])[0]
        body = body if isinstance(body, str) else "0x"
    sels = selectors(body)
    dec = int(r[3], 16) if ok(r[3]) and r[3] not in ("0x", "") else None
    owner = ("0x" + r[7][-40:]) if ok(r[7]) and len(r[7]) >= 42 else None
    owner_live = OWNER_SEL in sels and owner is not None and int(owner, 16) not in DEAD
    supply = int(r[6], 16) if ok(r[6]) and r[6] not in ("0x", "") else None
    sym, nm = _str(r[4]), _str(r[5])
    accepted, reason, powers = judge(sels, owner_live=owner_live, upgradeable=upgradeable, decimals=dec, symbol=sym, name=nm)
    return {"ok": accepted, "reason": reason, "powers": powers, "decimals": dec, "symbol": sym, "name": nm, "supply_raw": supply, "impl": impl,
            "owner": owner, "owner_live": owner_live, "upgradeable": upgradeable, "taxes": bool(sels & TAX_SEL), "roles": bool(sels & ROLE_SEL),
            "code_len": (len(body) - 2) // 2, "n_selectors": len(sels)}
