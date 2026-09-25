/** Audit of amount parsing/formatting (2026-09-25). "AUDIT FINDING" tests were failing before the 2026-09-25 fixes and now guard them. */
import { formatUnits, parseUnits } from 'viem'
import { describe, expect, it } from 'vitest'
import { parseAmount, token, tokenFloat } from './format'

const E18 = 10n ** 18n

describe('parseAmount rejects everything that is not a plain positive decimal', () => {
  it.each(['1e1', '1E18', '0x10', 'Infinity', '-Infinity', 'NaN', '-1', '+1', '1.2.3', '.', '', '   ', '1 000', '١', '１', '0', '0.0', '000', '1_000', '--1', '0b1'])(
    '%j -> null',
    (s) => expect(parseAmount(s, 18)).toBeNull(),
  )
  it('trims whitespace', () => expect(parseAmount('  5 ', 18)).toBe(5n * E18))
  it('accepts a trailing or leading dot', () => {
    expect(parseAmount('5.', 18)).toBe(5n * E18)
    expect(parseAmount('.000000000000000001', 18)).toBe(1n)
  })
  it('refuses more fraction digits than the token has (never rounds a wei away or up)', () => {
    expect(parseAmount('1.0000000000000000001', 18)).toBeNull()
    expect(parseAmount('0.0000001', 6)).toBeNull()
    expect(parseAmount('0.000001', 6)).toBe(1n)
  })
  it('keeps full precision above 2^53 (no Number round trip)', () => {
    expect(parseAmount('123456789012345678901234567.123456789012345678', 18)).toBe(123456789012345678901234567123456789012345678n)
  })
  it('the Max buttons round-trip exactly: formatUnits(balance) parses back to balance, no dust, never above', () => {
    for (const bal of [1n, 999n, 10n ** 18n - 1n, 123456789012345678901234567n, 2n ** 255n]) {
      expect(parseAmount(formatUnits(bal, 18), 18)).toBe(bal)
    }
    for (const bal of [1n, 1_000_001n, 5_000_000n]) expect(parseAmount(formatUnits(bal, 6), 6)).toBe(bal)
  })
  it('Max of a zero balance gives "0", which is refused rather than sent', () => {
    expect(parseAmount(formatUnits(0n, 18), 18)).toBeNull()
  })
  it('thousands grouping with commas still works', () => {
    expect(parseAmount('1,000', 18)).toBe(1000n * E18)
    expect(parseAmount('1,234,567.5', 18)).toBe(parseUnits('1234567.5', 18))
  })
})

/*
 * AUDIT FINDING (medium): parseAmount strips every comma, so a decimal comma ("0,5", common in EU locales and what
 * inputmode="decimal" keyboards offer) is read as a digit separator: "0,5" -> 5, "1,25" -> 125. The page then quotes,
 * approves and swaps/locks 10x–100x what the person meant (capped only by their balance).
 */
describe('AUDIT FINDING: a decimal comma is never silently read as a thousands separator', () => {
  it('"0,5" is not 5 tokens', () => expect(parseAmount('0,5', 18)).not.toBe(5n * E18))
  it('"1,5" is not 15 tokens', () => expect(parseAmount('1,5', 18)).not.toBe(15n * E18))
  it('"1,25" is not 125 tokens', () => expect(parseAmount('1,25', 18)).not.toBe(125n * E18))
  it('misplaced grouping ("12,34,5") is refused', () => expect(parseAmount('12,34,5', 18)).toBeNull())
})

describe('token() display', () => {
  it('truncates (never rounds up) so a displayed minimum is never more than what is enforced', () => {
    expect(token(999_999_999_999_999_999n, 18, 2)).toBe('0.99')
    expect(token(1_999_999n, 6, 6)).toBe('1.999999')
    expect(token(1_9999999n, 6, 6)).toBe('19.999999')
  })
  it('handles negatives, garbage and huge values', () => {
    expect(token(-1_500_000_000_000_000_000n)).toBe('−1.5')
    expect(token('not a number')).toBe('—')
    expect(token('0x10', 0)).toBe('16') // BigInt accepts hex strings: fine for relay wei strings, noted
    expect(token(2n ** 200n, 18, 0)).toMatch(/^[\d,]+$/)
  })
  it('tokenFloat is only approximate (USD maths), confirming nothing on-chain may use it', () => {
    expect(tokenFloat(123456789012345678901234567n)).toBeCloseTo(123456789.0123456789, 3)
  })
})
