# V8 revision notes (2026-10-29)

Draft: `lc_v8_10292026.tex` → `lc_v8_10292026.pdf` (32 pp.; V7 was 30).
Source of record: `../V7/lc_v7_10232026.tex`, as submitted on 2026-10-23. This revision responds to the SCIRP editor's comments of 2026-10-29 (`../V7/SCIRP Editor Comments - 10292026.docx`).
The V7 notes (`../V7/REVISION_NOTES.md`) still describe the data, journal format, reference audit and V6 corrections. Only the changes are listed here.

## Build

Unchanged from V7, except for the file names:

```bash
cd analysis
python extract_tables.py && python extract_runs.py && python analyze.py && python tables.py && python figures.py
cd .. && latexmk -pdf lc_v8_10292026.tex
```

The build has no overfull or underfull boxes, no undefined references and no BibTeX warnings. The abstract, keywords and title note still fit on page 1.

## Response to the editor

**1. Cost recommendations qualified.**
- §3.5 now states that every cost ratio and saving in the article is a classifier-fitting cost or saving, and that no deployment cost is measured.
- §4.3: the claim that excluding inference makes the decision rule "conservative" is removed.
  - The section now says its savings are fitting-time savings.
  - The cheaper runs recommended on 20 Newsgroups, IMDB and ArXiv-10 use frozen transformer embeddings. On IMDB and ArXiv-10 these come from the same model as the best fine-tuned run, and on 20 Newsgroups from Llama-3.2-1B, close in size to the DeepSeek model it replaces.
  - These configurations therefore need transformer inference on every new document, so their deployment savings are unmeasured. Only the TF-IDF, LSA and static-vector entries avoid it.
  - "Decision rule" is now "reads the frontier at fixed tolerances".
- §5.1:
  - The procedure opens with both qualifications: fitting time only, and test-set maxima.
  - Its items are reported as what was observed ("came within", "added ... here"), not as what to expect.
  - The retraining paragraph now notes that inference also recurs and that, for the frozen-transformer configurations, the deployment advantage may be considerably smaller than the fitting-time ratio.
- The same qualification is added to the abstract, Intro findings 2–3, the tolerance-table caption, Limitations ("What is timed") and the Conclusion.

**2. Selection of winners.**
- A new §3.6 paragraph, "Selection on test scores", states the following:
  - Every winner is selected on test scores. The classical module has no validation split, and the neural modules use theirs only for early stopping within a run; the run logs hold `final-te-*` measures only.
  - The paragraph lists every selection this affects.
  - It calls the results descriptive test-set maxima, optimistic as estimates for a configuration chosen in advance, more so for the classical family with its larger search.
  - Used as prospective guidance, they identify candidates to validate, not scores to expect.
- Also added:
  - a note after the Intro findings;
  - "best test-set run" in the abstract;
  - the peak and tolerance table captions;
  - a Limitations bullet, "Test-set selection";
  - a new extension in the Conclusion: validation-based selection and repeated seeds.

**3. Dependence in the class-embedding tests (new analysis).**
- `analyze.py` (`corpus_test`, `summarise_pairs`): each corpus's paired differences are averaged, and the exact two-sided Wilcoxon test is run over the eight per-corpus means, as in the family comparisons. Pair counts and pair medians are kept as descriptive statistics.
- `results.json` also keeps the per-corpus-median variant and the old pair-level $p$ (`p_corpus_median`, `p_pairs`).
- Table 10 gains "corpora up/down" columns, and its $p$ is now the corpus-level value.

| Comparison (Macro-F1) | Pairs up/down | Corpora up/down | V7 $p$ (pairs) | V8 $p$ (corpora) |
|---|---|---|---|---|
| WCE, static NNs | 80/45 | 5/3 | 1.5e-6 | **0.31** |
| Classical *dot* + WCE | 153/3 | 8/0 | 6e-27 | 0.008 |
| Classical *solo* + Π | 118/37 | 6/2 | 1e-13 | **0.055** (0.039 with medians) |
| Classical *lsa* + Π | 80/75 | 5/3 | 0.21 | 0.46 |
| Classical *cat-doc* + Π | 41/102 | 1/7 | 1e-12 | 0.016 |
| TCE, transformers | 39/35 | 5/3 | 0.66 | 0.55 |

**Consequence:** the WCE gain in static networks is no longer significant. It is positive on five corpora and concentrated on arXiv (+8.6) and Reuters-21578 (+4.8). The finding "WCEs help static NNs" is rewritten throughout:
- abstract;
- Intro finding 5 and contribution (iv);
- §4.6;
- §5.1 item 4;
- Conclusion.

The Moreo et al. sentence now says the positive pair-level median is consistent with their gains, but that the gain does not hold across corpora here.

**4. Nonsignificance vs. equivalence.**
- "TCEs have no effect" is replaced by "no statistically detectable systematic effect was found in these runs" in the abstract, §4.6 and the Conclusion. The same applies to the Intro (finding 5, contribution (iv), relation to the earlier version), §5.1 and the *lsa* result.
- §4.6 now says outright that a nonsignificant test does not establish equivalence. With eight corpora the test has little power, and per-corpus mean changes outside arXiv range from −1.1 to +0.7 points, so effects of about a point cannot be ruled out.
- The section title "Class embeddings: static networks yes, transformers no" is now "Class embeddings".
- "Neither help nor hurt" and "add nothing" are removed.

**5. WCE attribution.**
- §3.3: after the feature table, the text notes that $\Pi_{M+\mathrm{WCE}}(d) = x_d E_M \oplus x_d W$, so only *dot-wce* differs from its base form by the WCE columns alone. Confirmed from the logged dimensions: BBC/BERT *dot* has 768 columns and *dot-wce* 773, while *solo*, *lsa* and *cat-doc* each gain 773.
- §4.6 (classical): the *dot* result is attributed to the WCEs. The *solo*, *lsa* and *cat-doc* results are described as adding projected embedding features and WCEs together, and the section says these runs cannot separate the two.
- The "supplement for weak representations" conclusion and the "lexical evidence the WCEs summarise" explanation are removed.
- Table 10 and Figure 8 rows are relabelled "Classical, *solo* + $\Pi_{M+\mathrm{WCE}}$" and so on, and both captions are updated.
- §3.6 also notes that the *solo* comparison favours the *-wce* arm, which has 2–3 runs per pair against 1.

**6. 3,725 vs 3,724 runs.** The editor is right.
- The full run log (`layercake.all.summary.2025-04-11.csv`, line 5748) records the IMDB naive Bayes run `tfidf[xlnet]` with `final-te-macro-f1` and nothing after it: no Micro-F1 and no accuracy, unlike every other run. The per-dataset file matches.
- §3.6 now says there are 3,725 runs with Macro-F1, of which 3,724 also have Micro-F1, and names the run.
- The Table A5 caption explains the 19 runs and the total of 3,724.
- `analyze.py` writes `record_counts` and `macro_only_runs` to `results.json`.
- No other number changes: all other tables are byte-identical to V7, and the V6 verification still passes 784/784.

## Other changes

- Data and Code Availability now points to `paper/scirp/V8`.
- Acknowledgements: AI-assisted work is now dated 23 and 29 October 2026, and its scope includes the revision in response to editorial review. **The author should confirm this statement before resubmission.**

## Not yet done

- No new submission zip. Build one when the text is approved, as for V7.
