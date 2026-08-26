# Which languages actually work

Not "did it produce files" — files are easy. The question this page answers is whether
the **audio and the text belong together**, whether the text is in the **right language
and script**, and whether a **real training pipeline can load it**.

Ten languages were built into datasets from real YouTube audio (4-minute excerpts,
`--lang <code> --model large-v3`) and measured end to end. Every number here was measured
on the produced files — none is quoted from hearsay's own build report.

This sweep is what produced the five fixes in [0.7.0](../CHANGELOG.md).

## Results

| language | clips | pairing mean / median | mispaired control | script | FLEURS WER | trainable |
| --- | --- | --- | --- | --- | --- | --- |
| Vietnamese `vi` | 22 | **0.999** / 1.00 | 0.187 | 22/22 | 10.3 | **Yes** |
| Turkish `tr` | 22 | **0.995** / 1.00 | 0.247 | 22/22 | 8.4 | **Yes** |
| Japanese `ja` | 25 | **0.986** / 1.00 | 0.164 | 25/25 | 5.3 | **Yes** |
| Spanish `es` | 26 | **0.975** / 1.00 | 0.220 | 26/26 | 3.0 | **Yes** |
| Korean `ko` | 23 | **0.971** / 1.00 | 0.149 | 23/23 | 14.3 | **Yes** |
| Russian `ru` | 24 | **0.968** / 0.97 | 0.232 | 24/24 | 5.6 | **Yes** |
| Hindi `hi` | 24 | **0.917** / 0.94 | 0.223 | 24/24 | 21.5 | **Yes** |
| Mandarin Chinese `zh` | 28 | **0.909** / 0.94 | 0.088 | 28/28 | 14.7 | **Yes** |
| Swahili `sw` | 18 | **0.939** / 0.97 | 0.238 | 18/18 | 39.3 | **Marginal** |
| Bengali `bn` | 7 | **0.614** / 0.78 | 0.243 | 7/7 | 104.1 | **No** |

*FLEURS WER is Whisper's own word error rate for the language (its paper, Table 13,
`large-v2` — the last size the paper reports). It is the accuracy column; pairing is not.*

### What each verdict rests on

- **Vietnamese `vi` — Yes.** Diacritics survive slicing, csv, json and the zip byte-for-byte.
- **Turkish `tr` — Yes.** Agglutination broke 0 of 22 clip boundaries.
- **Japanese `ja` — Yes.** Kana in 25/25 clips — genuinely japanese, not han-only.
- **Spanish `es` — Yes.** With whisper. parakeet returned english — fixed in 0.7.0.
- **Korean `ko` — Yes.** 23/23 hangul; whisper handles the spacing, not hearsay.
- **Russian `ru` — Yes.** The one language of the ten `auto` still sends to parakeet, verified.
- **Hindi `hi` — Yes.** 24/24 devanagari; never came back as urdu.
- **Mandarin Chinese `zh` — Yes.** 28/28 han; clip cuts are not linguistic — see caveats.
- **Swahili `sw` — Marginal.** Real swahili, ~39% word error — stt weak supervision, not tts.
- **Bengali `bn` — No.** Bengali script but phonetic spelling; needs a fine-tune.

## How each column was measured

**Structure.** Every `manifest.jsonl` row resolves to a WAV that exists; the manifest
`duration` matches the decoded audio within 0.05 s (all ten matched exactly); mono,
16-bit, at the stated sample rate; no duplicate ids, no orphan WAVs, no empty text, no
control characters; every `metadata.csv` row has exactly three pipe-delimited fields and
carries the same text as the manifest. All ten passed every check.

**Pairing — does clip *N*'s audio go with clip *N*'s text?** Eight random clips per
language are re-transcribed with the **same model and language** and diffed against that
clip's own row. This isolates what hearsay controls (the pairing) from ASR accuracy,
which it does not.

A self-similarity score alone proves nothing: in a language the model transcribes badly
it is low even when the pairing is perfect. So every sample is **also** scored against a
*different* clip's text — a mispaired control. The **gap** is the pairing signal. Here it
ranged from +0.37 (Bengali) to +0.82 (Chinese, Japanese, Korean); a broken pairing would
show no gap at all.

**Script authenticity.** The language's Unicode block, plus the characters it actually
needs — Devanagari matras, Turkish `ç ğ ı ş`, Vietnamese `ă â ê ô ơ ư đ`, kana for
Japanese, Bengali matras — plus function-word counts where script alone cannot separate
two languages (Swahili from English). This is the check that catches a model
*transliterating* rather than transcribing.

**Boundaries.** First and last 25 ms RMS against the clip's own body. Checked for
discriminability first: the peak-to-quiet-floor ratio was 28x–1131x across the ten, so no
source had a continuous music bed and the measure is meaningful. On audio with a bed it
is not, and should not be reported.

**A real trainer loads it.** Each language was also built with `--format hf --format
jsonl` and loaded with `datasets.load_dataset("audiofolder", ...)`. All ten: every row
decoded, every text paired to its own row.

## Caveats — what these numbers do not show

- **Pairing is not accuracy.** Swahili scores 0.939 and is still roughly 39% wrong
  word-for-word (`kwanjia` for `kwa njia`, `makau makuia` for `makao makuu`). The model is
  *consistently* wrong, which self-similarity cannot see. Read the FLEURS column for accuracy.
- **Chinese and Japanese clips are cut at arbitrary positions.** `large-v3` returned almost
  no punctuation for either — 3 punctuation characters in 1183 for Mandarin, 1 in 987 for
  Japanese (identical with `--no-vad`, so this is not hearsay's VAD) — and Whisper marks no
  word boundaries in these scripts. Clip boundaries are therefore driven by pauses alone.
  Audio and text still match; the cuts are simply not linguistic.
- **One excerpt per language.** Findings about *hearsay* generalise across languages;
  findings about a language's ASR quality are a single 4-minute sample.
- **`auto` opens `whisper-small`** for Chinese, Spanish, Japanese, Turkish, Vietnamese and
  Korean. These datasets used `large-v3`. On Mandarin, `small` also returned **Traditional**
  characters where `large-v3` returned Simplified — consistent within one build, but do not
  mix builds if you care about orthography.
- **`large-v3` beats the FLEURS column** for several languages, since the paper stops at
  `large-v2`. Bengali is the clearest case: 104.1 would imply nothing usable, and `large-v3`
  at least returns correct Bengali script.

## Sources

| language | video | excerpt |
| --- | --- | --- |
| `vi` | [VTV4 Bản tin thời sự tiếng Việt](https://www.youtube.com/watch?v=a2AHdERfzuQ) | 150–390s |
| `tr` | [TRT Haber Ana Haber Bülteni](https://www.youtube.com/watch?v=KrIUDwhMRlI) | 150–390s |
| `ja` | [NHK FM ラジオニュース](https://www.youtube.com/watch?v=ilyusPMjurE) | 150–390s |
| `es` | [Oso Trava CRACKS PODCAST (Canelo interview)](https://www.youtube.com/watch?v=dqnOeSE2ZMU) | 900–1140s |
| `ko` | [KBS 뉴스광장](https://www.youtube.com/watch?v=XoRuA8sGOSA) | 150–390s |
| `ru` | [Подкаст Лазерсон — интервью об экономике](https://www.youtube.com/watch?v=fGgUVNYEkvI) | 600–840s |
| `hi` | [Sansad TV हिंदी समाचार बुलेटिन](https://www.youtube.com/watch?v=_YsToUySo8A) | 150–390s |
| `zh` | [CCTV 新闻联播 (Mandarin news anchor)](https://www.youtube.com/watch?v=uMJdzppHOAU) | 150–390s |
| `sw` | [DW Kiswahili Habari za Ulimwengu](https://www.youtube.com/watch?v=nUtKwbHhkKk) | 150–390s |
| `bn` | [Independent Bulletin বাংলা সংবাদ](https://www.youtube.com/watch?v=CugyDd08dtY) | 150–390s |

Previously verified the same way and not re-tested here: English, Uzbek, Urdu, Arabic,
Pashto, Polish.
