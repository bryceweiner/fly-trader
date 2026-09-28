"""The vault's ETH pot (Robinhood Chain memecoin profits, paid in native ETH on Robinhood Chain to the locker's EVM
address). Idempotent and additive like vault/schema.py; applied after it (db/schema.apply_schema). Amounts in wei."""
from __future__ import annotations

ETH_DDL: list[str] = [
    """CREATE TABLE IF NOT EXISTS vault_eth_settlements (
        id bigserial PRIMARY KEY,
        settlement_id bigint NOT NULL UNIQUE REFERENCES vault_settlements(id),
        period_start timestamptz NOT NULL, period_end timestamptz NOT NULL,
        status text NOT NULL CHECK (status IN ('allocated', 'failed')),
        book text NOT NULL,
        native numeric(78,0), open_cost numeric(78,0), lots numeric(78,0), deposits numeric(78,0), withdrawals numeric(78,0), payouts numeric(78,0),
        realized numeric(78,0), allocated_before numeric(78,0), pot numeric(78,0), liquid_cap numeric(78,0), allocated numeric(78,0), carried numeric(78,0),
        earners int, inputs_sha256 text, created_at timestamptz NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS vault_eth_allocations (
        eth_settlement_id bigint NOT NULL REFERENCES vault_eth_settlements(id),
        evm text NOT NULL,
        weight numeric(100) NOT NULL,
        wei numeric(78,0) NOT NULL CHECK (wei >= 0),
        PRIMARY KEY (eth_settlement_id, evm))""",
    # a claim's ETH leg: the amount fixed with the SOL amount, then paid by the RH wallet (rh_txs row eth_tx_id)
    "ALTER TABLE vault_claims ADD COLUMN IF NOT EXISTS eth_wei numeric(78,0)",
    "ALTER TABLE vault_claims ADD COLUMN IF NOT EXISTS eth_status text NOT NULL DEFAULT 'none'",
    "ALTER TABLE vault_claims ADD COLUMN IF NOT EXISTS eth_tx_id bigint",
    "ALTER TABLE vault_claims ADD COLUMN IF NOT EXISTS eth_attempts int NOT NULL DEFAULT 0",
    "ALTER TABLE vault_claims ADD COLUMN IF NOT EXISTS eth_reason text",
    "ALTER TABLE vault_claims DROP CONSTRAINT IF EXISTS vault_claims_eth_status_check",
    "ALTER TABLE vault_claims ADD CONSTRAINT vault_claims_eth_status_check CHECK (eth_status IN ('none', 'owed', 'sending', 'paid', 'failed'))",
    # owed ETH per holder: allocated − paid − in flight
    """CREATE OR REPLACE VIEW vault_eth_accounts AS
        SELECT a.evm, a.allocated, COALESCE(c.claimed, 0)::numeric(78,0) AS claimed, COALESCE(c.in_flight, 0)::numeric(78,0) AS in_flight,
               (a.allocated - COALESCE(c.claimed, 0) - COALESCE(c.in_flight, 0))::numeric(78,0) AS owed
        FROM (SELECT evm, SUM(wei)::numeric(78,0) AS allocated FROM vault_eth_allocations GROUP BY evm) a
        LEFT JOIN (SELECT evm,
                          SUM(eth_wei) FILTER (WHERE eth_status = 'paid') AS claimed,
                          SUM(eth_wei) FILTER (WHERE eth_status IN ('owed', 'sending')) AS in_flight
                   FROM vault_claims GROUP BY evm) c USING (evm)""",
]
