# Memo: Phoenix Suns shot profile, 2025-26

**To:** Basketball Operations · **Re:** Where the half-court offense loses value, and what
is noise vs. signal · **Data:** every 2025-26 regular-season FGA (219,160 shots), scored by
an xFG model trained only on the prior three seasons.

## Bottom line

1. **The offense's gap is shot *quality*, not shot-*making*.** PHX generated 1.082 expected
   points per shot (20th). It made shots at roughly the expected rate (−0.8 per 100, 17th).
   Better shooting luck would not have closed the gap.
2. **The biggest lever is rim attempts.** PHX took 23.4% of shots at the rim vs a 28.1%
   league median, 28th in the NBA. It was 30th, 25th and 30th the three prior seasons,
   across different rosters and staffs. Moving 5% of attempts from mid-range and paint
   floaters to the rim is worth **≈ +2.5 points per 100 shots, ≈ +2.3 points per game**. That
   would move PHX from 20th to about 11th in shot quality. This understates the gain,
   because rim attempts also draw the most free throws.
3. **The defense's results were a little better than its process.** It ranked 17th in shot
   quality allowed and 12th in opponent shot-making. Opponent shot-making is the least
   repeatable component in the league (half-season r = 0.48 vs 0.75 for quality allowed),
   so plan around the 17th.

## Evidence

| | Value | Rank (of 30) |
|---|---|---|
| Offense: expected points per shot | 1.082 | 20 |
| Offense: shot-making, per 100 shots | −0.8 | 17 |
| Defense: expected points per shot allowed | 1.099 | 17 |
| Defense: opponent shot-making, per 100 | −1.7 | 12 |

**Shot diet vs. league (share of attempts):**

| Zone | PHX | NBA | PHX xPPS |
|---|---|---|---|
| Rim | 23.4% | 28.4% | 1.37 |
| Paint (non-RA) | 19.1% | 20.0% | 0.89 |
| Mid-range | 12.2% | 10.1% | 0.84 |
| Corner 3 | 10.2% | 10.8% | 1.16 |
| Above-break 3 | 35.1% | 30.8% | 1.06 |

*(NBA column is the league-wide share. The 28.1% in the bottom line is the median team's
rim share.)*

<img src="shot_diet_PHX.png" width="460" alt="PHX hex map vs league: deep blue at the rim (fewer attempts), red in the short paint and above the break">

## Players: what is real

Shot-making below is points above expected per 100 shots. Raw numbers are shrunk toward the
league mean in proportion to their noise; leaguewide, it takes ~272 FGA (~330 threes)
before an estimate is half signal.

<img src="players_PHX.png" width="520" alt="Suns players' raw and shrunk shot-making with 90% intervals">

- **Devin Booker: the diet only works if the making is elite.** About 53% of his
  attempts are non-rim twos, and just 16% come at the rim, so his shot quality is low
  (1.00 xPPS vs a 1.08 team average). He still beats expectation on those twos (45.8% on
  mid-range vs 42.2% expected). His overall shot-making has fallen from +9.1 and +9.8 per 100
  (2022-24) to +4.5 and now +1.2. The 2023-24 to 2025-26 gap is about two standard errors:
  probably real, not certainly. If it is real, the offense loses the cushion that let a
  low-xPPS diet work. **Action:** treat Booker's rim share (and the actions that create it)
  as a design target, not a by-product, and review the 2025-26 three-point dip on film
  before treating it as decline.
- **Jalen Green: consistent under-conversion, good shots.** He gets good looks (1.09 xPPS)
  but has finished below expectation four straight seasons (−7.4, −7.1, −4.3, −11.1). In
  2025-26 it was at the rim (60% vs 69% expected) and on above-break threes (28% vs 36%).
  It is the only Suns 90% interval entirely below zero. A four-year pattern is a player
  trait, not variance: **the finishing mechanics at the rim and the above-break three are
  the development targets**, and his projections should not assume a bounce-back to
  league-average making.
- **Collin Gillespie and Royce O'Neale: positive, but don't pay for it yet.** Both are
  +8 to +10 raw. Shrunk, they are +6, and both intervals still touch zero. Gillespie has been
  positive in all three seasons on small samples, which is encouraging. The multi-season
  model below would firm this up.
- **Mark Williams and Oso Ighodaro** convert at expectation on high-quality looks (1.28–1.29
  xPPS). They take 70% and 61% of their shots at the rim, the shot type the diet needs more of.

## Confidence and caveats

- **High confidence:** the rim-share pattern (four seasons, full-season samples) and the
  team quality ranks.
- **Moderate:** individual shot-making for players under ~500 FGA, which is why it is shown
  with intervals.
- **Not modeled:** defender distance and tracking context (not in public data), free throws,
  and whether the personnel can produce more rim pressure. The +2.3 points per game prices
  shot *selection* at current efficiency. It is an upper-bound-style estimate for the shot
  mix, not a forecast.

## Suggested next steps

1. Rebuild xFG with tracking features (closest defender, touch time, dribbles) to separate
   contested from open looks in the same zone.
2. Use a multi-season, role-aware prior for shot-making, which matters most for Green,
   Gillespie and Booker, where several seasons point the same way.
3. Break rim-attempt generation down by action (drives, cuts, rolls, transition) and by
   lineup, to find which units and sets produce rim pressure.
