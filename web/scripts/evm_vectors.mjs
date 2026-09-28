// Reference vectors for the Python EVM layer (fly_trader/rh/tx.py, rh/abi.py), produced by viem: signed EIP-1559
// transactions and ABI encodings. Regenerate: node web/scripts/evm_vectors.mjs > tests/vectors/evm_tx_v1.json
import { privateKeyToAccount } from 'viem/accounts'
import { encodeAbiParameters, parseAbiParameters, keccak256, serializeTransaction, encodeFunctionData, parseAbi, toFunctionSelector } from 'viem'

const keys = [
  '0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318',
  '0x0000000000000000000000000000000000000000000000000000000000000001',
  '0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140',
]
const txs = [
  { chainId: 4663, nonce: 0, maxPriorityFeePerGas: 0n, maxFeePerGas: 20000000n, gas: 21000n, to: '0x000000000000000000000000000000000000dEaD', value: 1n, data: '0x' },
  { chainId: 4663, nonce: 127, maxPriorityFeePerGas: 1n, maxFeePerGas: 3000000000n, gas: 350000n, to: '0x8876789976decbfcbbbe364623c63652db8c0904', value: 10n ** 16n, data: '0x3593564c' + 'ab'.repeat(200) },
  { chainId: 46630, nonce: 1024, maxPriorityFeePerGas: 1000000n, maxFeePerGas: 1000000n, gas: 60000n, to: '0x5fc5360D0400a0Fd4f2af552ADD042D716F1d168', value: 0n, data: '0x095ea7b3' + '00'.repeat(64) },
  { chainId: 4663, nonce: 2 ** 32, maxPriorityFeePerGas: 0n, maxFeePerGas: 2n ** 64n, gas: 30000000n, to: '0x6131B5fae19EA4f9D964eAc0408E4408b66337b5', value: 2n ** 100n, data: '0x',
    accessList: [{ address: '0x8366a39cc670b4001a1121b8f6a443a643e40951', storageKeys: ['0x' + '11'.repeat(32), '0x' + '00'.repeat(31) + '01'] }] },
]
const out = { note: 'viem 2.56.8; fly_trader/rh/tx.py and rh/abi.py must reproduce every field', txs: [], abi: [] }
for (const k of keys) {
  const acct = privateKeyToAccount(k)
  for (const t of txs) {
    const tx = { type: 'eip1559', ...t }
    const raw = await acct.signTransaction(tx)
    out.txs.push({ key: k, address: acct.address.toLowerCase(), tx: { ...t, nonce: t.nonce, maxPriorityFeePerGas: t.maxPriorityFeePerGas.toString(), maxFeePerGas: t.maxFeePerGas.toString(),
      gas: t.gas.toString(), value: t.value.toString(), accessList: t.accessList || [] }, unsigned: serializeTransaction(tx), raw, hash: keccak256(raw) })
  }
}
const abiCases = [
  ['uint256,address,bool', [123456789n, '0x8366a39cc670b4001a1121b8f6a443a643e40951', true]],
  ['int24,int24,uint128', [-887200, 887200, 2n ** 127n]],
  ['bytes,string,bytes32', ['0x' + 'ab'.repeat(33), 'fly trader', '0x' + '01'.repeat(32)]],
  ['address[],uint256[]', [['0x0000000000000000000000000000000000000001', '0x0000000000000000000000000000000000000002'], [1n, 2n, 3n]]],
  ['(address,address,uint24,int24,address),bool,uint128,uint128,bytes', [['0x0000000000000000000000000000000000000000', '0x1111111111111111111111111111111111111111', 0, 200, '0xe5e702641ea86f4ae6cc3cdaed2b886f976be044'], true, 10n ** 18n, 1n, '0x']],
  ['(address,(uint256,bytes)[],string)[]', [[['0x2222222222222222222222222222222222222222', [[1n, '0x01'], [2n, '0x0203']], 'a'], ['0x3333333333333333333333333333333333333333', [], '']]]],
]
for (const [types, values] of abiCases) {
  const enc = encodeAbiParameters(parseAbiParameters(types), values)
  out.abi.push({ types, values: JSON.parse(JSON.stringify(values, (_, v) => typeof v === 'bigint' ? v.toString() : v)), encoded: enc })
}
out.selectors = ['execute(bytes,bytes[],uint256)', 'approve(address,uint256)', 'swap((address,address,bytes,(address,address,address[],uint256[],address[],uint256[],address,uint256,uint256,uint256,bytes)))']
  .map(s => ({ sig: s, selector: toFunctionSelector('function ' + s) }))
console.log(JSON.stringify(out, null, 1))
