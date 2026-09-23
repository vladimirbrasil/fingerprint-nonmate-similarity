# How close do two different fingers get? A first measurement

Question behind this work (from a correspondence with Michael Levin, Sept 2026): *has a living
person's fingerprint ever matched a deceased person's record — and can the available databases be
searched for such a case?*

This repository answers the part that is measurable with public data: **how similar do two
fingerprints from different fingers actually get?** Everything here runs on public datasets and
public-domain software, on an ordinary desktop machine. No law-enforcement system is involved.

## Method

- **Matcher:** NIST NBIS — `mindtct` (minutiae extraction) + `bozorth3` (matching), public domain.
- **Data:** the openly downloadable "set B" of the Fingerprint Verification Competition databases
  (FVC2000, FVC2002, FVC2004; DB1–DB4), 960 images, 120 finger labels, 8 impressions each.
- **Comparison:** all against all — 460,320 pairs (3,360 same-finger, 456,960 different-finger).
- **Hardware:** Intel Celeron N5105, 4 cores. Extraction 75 ms/image; matching 579 pairs/s/core.
  The whole run took 158 seconds.

## Result 1 — the two distributions barely touch

| Percentile | Same finger | Different fingers |
|---|---|---|
| 50 | 58 | 6 |
| 90 | 150 | 10 |
| 99 | 229 | 17 |
| 99.9 | 292 | 30 |
| max | 366 | 59 (corrected — see below) |

Half of all different-finger pairs score 6 or less on a scale where same-finger pairs sit around 58.
At the operational threshold of 40 often used with `bozorth3`, **2 different-finger pairs out of
456,960 cross the line (0.0004%)**, and the highest of them reaches 59 — a score that a same-finger
pair beats half the time.

## Result 2 — the interesting part: the "matches" that weren't

The naive run reported 357 different-finger pairs above 40, including 129 above 100 — scores in
genuine territory. Every one of the top 200 came from the same pair of databases (FVC2000 DB1 vs
DB2) and carried **the same finger index**. All ten indices, 101 to 110, show it, with scores from
40 to 200.

They are not coincidences. They are **the same fingers, captured on two different sensors**, which
the naive labelling counted as different people. A second, smaller case of the same artifact appears
between the synthetic databases (DB4) of 2000 and 2002: the generator reused identities.

After removing same-index cross-database pairs, the highest different-finger score drops from 200 to
**59**.

This matters more than the number. The one thing in this data that looked like "two different people
with the same fingerprint" was a **record duplication**. In a national database, that is exactly what
a hit against a person who is supposed to be dead would look like first — and exactly what would be
closed as a clerical error, correctly in almost every case, without anyone recording it.

## Result 3 — the closest non-mate pair, for a human to judge

`out/figs/par_mais_proximo.png` shows the two fingers that scored 59: FVC2002 DB3 finger 108,
impression 7, and FVC2004 DB3 finger 102, impression 1. To an examiner they are plainly different
fingers — different pattern class, different ridge flow. The algorithm's best "near miss" in almost
half a million comparisons does not survive one second of human inspection.

## Limits (stated plainly)

- 120 fingers is a small population. The tail of this distribution is what matters, and a small
  sample under-samples tails. The next step is SOCOFing (600 subjects, 6,000 images, ~18M pairs;
  about 3 hours on this machine) and, with a data request, the NIST SD300–303 series.
- One matcher. SourceAFIS should be run as an independent second opinion; agreement between two
  unrelated algorithms is what separates a property of fingerprints from a property of `bozorth3`.
- These are flat, good-quality impressions. Real forensic errors (Mayfield 2004, McKie 1997) happen
  on partial, distorted latent marks, where the information available is a fraction of this.
- A high score is not an identification. Identification is a human decision (ACE-V); this is a
  similarity measurement.

## Reproduce

```bash
scripts/00-build-nbis.sh     # compiles mindtct + bozorth3 from NIST sources
scripts/01-bench.sh          # measures speed on your machine
scripts/10-download-fvc.sh   # fetches the public FVC "set B" databases
scripts/20-extract.sh        # minutiae for every image
scripts/30-allvsall.py       # all-against-all, histograms + top impostor pairs
```

Outputs land in `out/`: `summary.json`, `hist_*.csv`, `top_impostors.csv`, `corrigido.json`.
