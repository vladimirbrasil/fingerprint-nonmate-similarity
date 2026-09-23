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
| max | 366 | 39 (after removing the duplicates of Results 2 and 3) |

Half of all different-finger pairs score 6 or less on a scale where same-finger pairs sit around 58.
At the operational threshold of 40 often used with `bozorth3`, 2 different-finger pairs out of
414,720 cross the line, once the 42,240 same-index cross-database pairs are set aside (Result 2
explains why). **Both turned out to be the same finger** (Result 3). With that duplicate removed,
**no pair of genuinely different fingers reaches 40**: the highest of 414,656 scores 39.

## Result 2 — the interesting part: the "matches" that weren't

The naive run reported 357 different-finger pairs above 40, including 129 above 100 — scores in
genuine territory. Every one of the top 200 came from the same pair of databases (FVC2000 DB1 vs
DB2) and carried **the same finger index**. All ten indices, 101 to 110, reach at least 40, with peaks up to
200. Most of them score like same-finger pairs throughout; two do not (see Result 5).

They are not coincidences. They are **the same fingers, captured on two different sensors**, which
the naive labelling counted as different people. A second, smaller case of the same artifact appears
between the synthetic databases (DB4) of 2000 and 2002: the generator reused identities.

After removing same-index cross-database pairs, the highest different-finger score drops from 200 to
**59**.

This matters more than the number. The one thing in this data that looked like "two different people
with the same fingerprint" was a **record duplication**. In a national database, that is exactly what
a hit against a person who is supposed to be dead would look like first — and exactly what would be
closed as a clerical error, correctly in almost every case, without anyone recording it.

## Result 3 — the closest "different" pair was the same finger

The pair that scored 59 with `bozorth3` joins FVC2002 DB3 finger 108 and FVC2004 DB3 finger 102,
two collections made two years apart on different sensors (capacitive in 2002, thermal sweep in
2004). SourceAFIS independently ranks the same finger pair first among 6,480 (Result 4).

**It is one finger.** A fingerprint examiner (the author)
compared all 64 impression pairs and found at least 12 corresponding minutiae in every one. The same
dermal scars appear in both collections, and where pores are visible they sit in corresponding
positions. The volunteer was enrolled in both collections under different labels, and nothing in the
data links the two. An earlier draft of this report called the pair "different pattern classes";
that was wrong, and it was written before anyone had looked properly. Blind verification by a second
examiner has not yet been done.

The matchers mostly missed it. Of the 64 comparisons, 62 score below 40 in `bozorth3` (median 14)
and 60 in SourceAFIS (median 12.8). The small, partly overlapping capture areas of the two sensors
are the likely reason.

Figure: `results/par_108x102_dois_matchers.png`. The point-by-point figure will be added with the
examiner's marked correspondences.

## Result 4 — a second, unrelated matcher (SourceAFIS)

The same 960 images went through SourceAFIS 3.18.1 (open source, different extraction and matching
from NBIS). Extraction and all 921,600 directed comparisons took 70 seconds on the same machine.
`scripts/31-nbis-matrix.py` then writes the full `bozorth3` matrix so the two can be compared pair
by pair. The known cross-database duplicates are excluded from the comparison; 414,720 genuinely
different-finger pairs remain.

| | NBIS (bozorth3) | SourceAFIS |
|---|---|---|
| Same finger, median | 58 | 76.6 |
| Different fingers, median / p99.9 / max | 6 / 23 / 59 | 1.5 / 24.5 / 53.8 |
| Different-finger pairs ≥ 40 | 2 | 10 |
| Rank correlation, different-finger pairs | 0.28 | |

**The two algorithms disagree about which *impressions* are closest.** Their top 10 different-finger
pairs share none, and their top 100 share 2. No single pair falls in the extreme 0.01% of both. The
highest `bozorth3` pair scores 13 in SourceAFIS; the highest SourceAFIS pair scores 10 in `bozorth3`.
At the level of individual extreme scores, the tail describes the algorithm more than the finger.

**They agree much better about which *fingers* are closest.** Take the median over the 64 impression
pairs of each finger pair: the rank correlation rises to 0.59. The same finger pair (FVC2002 DB3/108
× FVC2004 DB3/102) sits in the top 10 of 6,480 for both algorithms. Its impressions are close
throughout, and each algorithm picks a *different* impression pair as its peak. Several other top
finger pairs come from the synthetic databases (DB4), which may reflect the generator's shared priors
rather than anything about real skin.

Practical consequence for any search of the kind the question asks for: a single high score from a
single algorithm is not a signal. A candidate should hold up across algorithms **and** across
impressions before a human looks at it.

## Result 5 — a duplicate can also hide as a non-match

The FVC2000 DB1×DB2 duplicates (Result 2) serve as a control: the same fingers on two sensors. Most
of them score like same-finger pairs in both algorithms. Two do not. Finger 108 has a `bozorth3`
median of 12 and a SourceAFIS median of 2.7; finger 110 has 20.5 and 0.0. Those values fall inside
the range of the closest *different*-finger pairs. Low-quality impressions push genuine pairs down as
well: in FVC2002 DB3, the weakest finger's same-finger median is 16.5 in `bozorth3`.

The error therefore runs both ways. A duplicated record can look like a coincidence (Result 2). It
can also look like two different people (this result). For the original question this is the harder
half: a real "same pattern in another body" would first have to be told apart from a record that
nobody linked, captured on another device, years apart. The FVC2002 DB3 × FVC2004 DB3 pair above is
exactly this situation, and Result 3 shows how it resolved: the same finger, labelled as two
people, scored like strangers by both algorithms.

Figure: `results/par_108x102_dois_matchers.png` shows the peak pair of each algorithm for this finger
pair. The full numbers are in `results/comparacao_matchers.json`.

## Result 6 — SOCOFing, 18 million pairs (preliminary)

SOCOFing (Shehu et al., 2018; 600 subjects, 10 fingers each, one impression per finger) was run
all-against-all with both matchers: 17,997,000 pairs. Measured ridge period puts the images' real
resolution near 160 dpi rather than 500; they were upsampled 3x before matching. A sanity test
(each "Altered-Easy / central rotation" image searched against 600 real fingers) found the right finger
first 99.8% of the time with `bozorth3` and 99.7% with SourceAFIS. The altered images take no part in
the statistics.

**The extreme tail is, again, records rather than fingers.** The dataset contains 11 groups of
byte-identical images filed under different labels (26 files; e.g. one image filed as the left little
finger of five different subjects), and subjects 596 and 598 behave like one person (or one person
with shuffled labels). After removing byte-identical copies, 41 pairs still score ≥ 40 in *both*
matchers. Each needs an examiner: 29 involve 596/598, and 12 are two differently-named fingers of the
same subject. The first of them, examined visually, is the same capture filed as two fingers. Until
that review is done, no maximum from this dataset is reported. The raw maximum (SourceAFIS 1,178)
is a duplicated file.

**`bozorth3` is not usable alone on these images.** It puts 13,499 different-finger pairs at ≥ 40
against SourceAFIS's 30: upsampled low-resolution images give spurious minutiae, and `bozorth3` counts them.

**Same body vs. different bodies (first look).** Different fingers of the same person score slightly
higher than fingers of different people across the upper distribution (99th percentile: 30 vs 23 in
`bozorth3`, 13.9 vs 11.6 in SourceAFIS; medians essentially equal). A finger and its mirror-twin on the
other hand (e.g. left vs right index) show no extra similarity at the 99th percentile. This is
consistent with the known correlation of pattern class within a person. It is preliminary, because
mislabelled duplicates within subjects inflate exactly this group's extreme tail.

Numbers: `results/socofing_analise.json`; byte-identical groups: `results/socofing_duplicatas_exatas.json`.

## Limits (stated plainly)

- 120 fingers in FVC, 6,000 in SOCOFing (Result 6). Both are small next to national databases, and
  SOCOFing's images are low-resolution. The NIST SD300–303 series (by data request) is the next step.
- Two matchers now (Result 4). Both are open-source, not a commercial AFIS. Commercial matchers
  are stronger, and their tails may differ again.
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
scripts/31-nbis-matrix.py    # full bozorth3 matrix
(cd sourceafis && mvn package) && java -cp 'sourceafis/target/*:sourceafis/target/lib/*' AllVsAll .
scripts/32-compare-matchers.py  # NBIS vs SourceAFIS, pair and finger level
```

Outputs land in `out/`: `summary.json`, `hist_*.csv`, `top_impostors.csv`, `corrigido.json`.
