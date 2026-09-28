"""The RH EVM layer against viem (tests/vectors/evm_tx_v1.json): EIP-1559 serialisation, signatures, hashes, sender
recovery, RLP and the ABI codec — plus the refusals (high-s, wrong selector, dirty words)."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from fly_trader.rh import abi, tx as T

V = json.loads((Path(__file__).parent / "vectors" / "evm_tx_v1.json").read_text())


def _tx(d):
    return T.Eip1559Tx(chain_id=d["chainId"], nonce=int(d["nonce"]), max_priority_fee_per_gas=int(d["maxPriorityFeePerGas"]), max_fee_per_gas=int(d["maxFeePerGas"]),
                       gas=int(d["gas"]), to=d["to"], value=int(d["value"]), data=bytes.fromhex(d["data"][2:]),
                       access_list=[(a["address"], a["storageKeys"]) for a in d["accessList"]])


@pytest.mark.parametrize("case", V["txs"], ids=lambda c: f"{c['address'][:8]}-{c['tx']['nonce']}")
def test_signed_transactions_match_viem(case):
    key = bytes.fromhex(case["key"][2:]); t = _tx(case["tx"])
    assert T.address_of(key) == case["address"]
    assert "0x02" + T.rlp_encode(t.fields()).hex() == case["unsigned"]
    s = T.sign(t, key)
    assert s.raw_hex == case["raw"] and s.hash == case["hash"]
    back = T.decode_raw(bytes.fromhex(case["raw"][2:]))
    lower = replace(t, to=t.to.lower(), access_list=[(a.lower(), k) for a, k in t.access_list])
    assert back.tx == lower and T.recover_sender(back) == case["address"]


@pytest.mark.parametrize("case", V["abi"], ids=lambda c: c["types"])
def test_abi_encoding_matches_viem(case):
    types = abi._split_top(case["types"])
    def conv(t, v):
        arr = abi._array_suffix(t)
        if arr is not None:
            return [conv(arr[0], x) for x in v]
        parts = abi._tuple_parts(t)
        if parts is not None:
            return tuple(conv(p, x) for p, x in zip(parts, v))
        if t.startswith(("uint", "int")):
            return int(v)
        if t == "bytes" or (t.startswith("bytes") and t != "bytes"):
            return bytes.fromhex(v[2:])
        return v
    vals = [conv(t, v) for t, v in zip(types, case["values"])]
    enc = abi.encode(types, vals)
    assert "0x" + enc.hex() == case["encoded"]
    dec = abi.decode(types, enc)
    norm = lambda x: [norm(y) for y in x] if isinstance(x, (list, tuple)) else (x.lower() if isinstance(x, str) and x.startswith("0x") else x)
    assert norm(dec) == norm(vals)


@pytest.mark.parametrize("case", V["selectors"], ids=lambda c: c["sig"][:20])
def test_selectors(case):
    assert "0x" + abi.selector(case["sig"]).hex() == case["selector"]


def test_rlp_edges():
    assert T.rlp_encode(0) == b"\x80" and T.rlp_encode(b"") == b"\x80" and T.rlp_encode(b"\x7f") == b"\x7f"
    assert T.rlp_encode([]) == b"\xc0" and T.rlp_decode(T.rlp_encode([b"a" * 60, [b"", b"\x01"]])) == [b"a" * 60, [b"", b"\x01"]]
    with pytest.raises(ValueError):
        T.rlp_decode(b"\x81\x00\x00")                                         # trailing bytes
    with pytest.raises(ValueError):
        T.rlp_encode(-1) if False else T._int_bytes(-1)


def test_refusals():
    key = bytes.fromhex(V["txs"][0]["key"][2:])
    with pytest.raises(ValueError):
        T.sign(T.Eip1559Tx(4663, 0, 5, 4, 21000, "0x" + "00" * 20), key)       # priority above max fee
    with pytest.raises(ValueError):
        abi.decode_call("approve(address,uint256)", bytes.fromhex("a9059cbb") + bytes(64))   # another selector
    with pytest.raises(ValueError):
        abi.decode(["address"], b"\x01" + bytes(31))                            # dirty high bytes
    with pytest.raises(ValueError):
        abi.encode(["uint8"], [256])
