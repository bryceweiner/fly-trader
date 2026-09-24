"""The $FLY vault: the hosted fly's side of profit sharing (docs/vault/SPEC.md).

Holders lock $FLY in FlyVault on Robinhood Chain; this package keeps the SOL ledger of the fly's trading wallet
(deposits, withdrawals, claims, realized profit R), settles the week's pot by time-weighted locked $FLY, pays claims
signed by both the holder's EVM and Solana wallets, and pushes the public stats to the relay on the site. Everything
here is off unless ``VAULT_ENABLED``; the Mac research fly never runs it.
"""
