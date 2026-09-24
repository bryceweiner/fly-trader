"""The two texts a vault claim is signed over (docs/vault/SPEC.md §2).

Both are rendered here from the claim's fields and never taken from the relay: a compromised relay can drop or replay a
claim but cannot make the fly verify text it did not render. The site renders the same bytes (web/src/lib/claim.ts);
tests/vectors/claim_v1.json pins both renderers.
"""
from __future__ import annotations

import calendar
import re
import time
from dataclasses import asdict, dataclass

from . import evm

REQUEST_ID = "fly-vault-claim-v1"
TTL_S = 900
NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
RFC3339_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SOL_CHAINS = ("mainnet", "devnet")
_B58 = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
_DOMAIN = re.compile(r"^[A-Za-z0-9.-]+(:\d{1,5})?$")


def rfc3339(ts: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(ts)))


def parse_rfc3339(s: str) -> int:
    if not RFC3339_RE.match(s or ""):
        raise ValueError("timestamp must be YYYY-MM-DDTHH:MM:SSZ")
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


@dataclass(frozen=True)
class ClaimFields:
    domain: str
    uri: str
    chain_id: int
    sol_chain: str
    evm: str          # any case in; rendered EIP-55
    sol: str
    nonce: str
    issued_at: str
    expires_at: str

    def check(self) -> None:
        """Syntax only; signatures, expiry against the clock and uniqueness are the caller's."""
        if not _DOMAIN.match(self.domain):
            raise ValueError("bad domain")
        if "\n" in self.uri or not self.uri.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("bad uri")
        if int(self.chain_id) <= 0:
            raise ValueError("bad chain id")
        if self.sol_chain not in SOL_CHAINS:
            raise ValueError("bad solana chain")
        if not evm.is_address(self.evm):
            raise ValueError("bad evm address")
        if not _B58.match(self.sol):
            raise ValueError("bad solana address")
        if not NONCE_RE.match(self.nonce):
            raise ValueError("bad nonce")
        if parse_rfc3339(self.expires_at) - parse_rfc3339(self.issued_at) != TTL_S:
            raise ValueError("expiry must be issue + 900 s")

    def as_dict(self) -> dict:
        return asdict(self)


def evm_text(f: ClaimFields) -> str:
    a = evm.to_checksum(f.evm)
    return (f"{f.domain} wants you to sign in with your Ethereum account:\n{a}\n\n"
            f"Pay the SOL the $FLY vault owes this address to Solana wallet {f.sol}.\n\n"
            f"URI: {f.uri}\nVersion: 1\nChain ID: {int(f.chain_id)}\nNonce: {f.nonce}\nIssued At: {f.issued_at}\n"
            f"Expiration Time: {f.expires_at}\nRequest ID: {REQUEST_ID}")


def sol_text(f: ClaimFields) -> str:
    a = evm.to_checksum(f.evm)
    return (f"{f.domain} wants you to sign in with your Solana account:\n{f.sol}\n\n"
            f"Receive the SOL the $FLY vault owes Ethereum account {a}.\n\n"
            f"URI: {f.uri}\nVersion: 1\nChain ID: {f.sol_chain}\nNonce: {f.nonce}\nIssued At: {f.issued_at}\n"
            f"Expiration Time: {f.expires_at}\nRequest ID: {REQUEST_ID}")
