# V7 revision notes (2026-10-23)

Draft: `lc_v7_10232026.tex` → `lc_v7_10232026.pdf` (30 pp. in the IJIS layout; V6 was 79 pp. as published).
Source of record for the text: `../V6/Layer Cake 07312025.tex`, the latest SCIRP revision (verified by timestamp and diff).

## Data used (no experiments re-run)

- **Run-level results** from the final analysis output behind the published paper (`out/20250411.lc_full_dataset_04102025.tsv/analysis_data/`, copied to `data/runs/`): 3,725 runs (3,197 classical, 303 static-embedding, 225 transformer), each with Macro/Micro-F1 and training time. Full-precision time and hardware are attached for 89% of runs from the full run log.
- **Verification:** aggregated per corpus and classifier, these runs reproduce **all 784** published summary values checked (max, mean, median, min, count) to three decimals.
- `out/20250427.ml_def_full_test.test` was inspected but not used: it is a later ML-only re-run whose counts differ from the published tables in two cells.
- The hyperbolic experiment is excluded entirely, as requested: no text, figures, data or references.

## Build

```bash
cd analysis
python extract_tables.py   # V6 published tables -> ../data/*.csv (used only for verification)
python extract_runs.py     # run-level results -> ../data/runs.csv
python analyze.py          # analyses -> ../data/derived/*.csv, results.json
python tables.py           # LaTeX table bodies -> ../tables/*.tex
python figures.py          # figures -> ../pictures/fig_*.pdf
cd .. && latexmk -pdf lc_v7_10232026.tex      # pdfLaTeX: free equivalents of the template fonts
#        latexmk -xelatex lc_v7_10232026.tex  # XeLaTeX: Minion Pro / Cambria / Calibri when installed
```

Use the base conda Python (`/opt/anaconda3/bin/python`). No number in a results table is typed by hand.

## Journal format (IJIS template)

The manuscript follows `../ijis-template.docx` (Word render: `../ijis-template.pdf`):

- **Class and page:** `ijis.cls` rebuilds the template's page, a 21 × 27.5 cm sheet with 1.5 cm side margins. Body text runs in a 12.71 cm column indented 5.29 cm, at 10 pt on an exact 15 pt line.
- **Page 1:** the SCIRP logo and journal block in the header, then the title with its note, authors, affiliation and email.
- **Side column and abstract:** the side column holds "How to cite", the dates, the copyright notice and the licence badges. Beside it sits the Abstract/Keywords panel between #943634 rules.
- **Headings and figures:** numbered headings ("1." / "1.1.") in #943634. Captions are labelled "Table 1." / "Figure 1." in bold accent colour (tables above, figures below), and in-text references to figures and tables appear in the same style.
- **Running head and footer:** the author header carries a grey rule. The footer has the DOI, page number and journal name.
- **Back matter:** Data and Code Availability, Author Contributions, Conflicts of Interest and References, as in the template.
- **Oversized tables:** these extend into the side column, as the template permits (model table, peak tables, tolerance table, class-embedding table).
- **References:** `scirp.bst` produces the template's format. Authors appear as "Last, F.M., ... and Last, F.M.", then the year, a title-case title, an italic journal, a bold volume and the pages, with the DOI or URL on its own line. References are numbered in order of citation.
- **Figures:** data figures are rendered at the 12.71 cm column width with 8 pt Times New Roman labels, as the template specifies. The diagrams use Times labels.
- **Fonts:**
  - The template asks for Minion Pro Capt (body), Cambria (headings) and Calibri/Tahoma (journal furniture).
  - **pdfLaTeX** uses the free equivalents from TeX Live: Crimson Pro, Caladea (metric-compatible with Cambria) and Carlito (metric-compatible with Calibri).
  - **XeLaTeX** uses Minion Pro, Cambria and Calibri when they are installed, and the free equivalents otherwise.
  - The three free families are also installed in `~/Library/Fonts` (OFL / Apache 2.0), so other applications can use them.
  - Minion Pro, Cambria and Calibri are licensed fonts and were not installed. To use Minion Pro, activate it in Adobe Creative Cloud (Adobe Fonts), then build with XeLaTeX. Cambria and Calibri ship with Microsoft Office under a licence limited to Office.
- **Template assets** (`ijis/`):
  - The SCIRP logo is from the template's embedded image.
  - The Open Access badge (and the template's CC BY badge, kept for reference) are cropped from a render of the template's first page.
  - The CC BY-NC button is the official Creative Commons button.
- **Licence:** CC BY-NC 4.0, as for the 2025 paper (per the signed copyright form, `../lc_scirp_copyright_form_06202025.pdf`).
- **Author Contributions:** P.J.W. is the main author: conceptualization, methodology, software, investigation (all experiments), formal analysis, data curation, visualization and writing (original draft). I.C. and P.J.W.: writing (review and editing).
- **Acknowledgements:** a section before Author Contributions discloses the use of Claude Code (Anthropic) with Claude Opus 5.5 (`claude-opus-5-5`) on 2026-10-23. It covers the redrafting, re-analysis and validation of results, reference checking, figures and tables, and typesetting.

## Title

**The Cost of Accuracy in Text Classification: Tradeoffs in Data Representation and Model Design for Classical ML, Neural Network, and Transformer-based Models** (chosen by the authors, 2026-10-23; replaces the working title "METRON").

- The title, PDF metadata and "How to cite" text carry it; the running head carries the authors, as in the template.
- At the template's 22 pt the title ran to four lines and pushed the title block off page 1. `ijis.cls` now sets it at 20/23 pt, which gives three lines and keeps the abstract, keywords and title note on page 1.
- The title note cites the 2025 paper as "a revised and extended version of [1]".

## Findings (all from the run-level data)

1. **Peak, per corpus.** Transformers lead the best classical run by 3.2 Macro-F1 points on average (V6 reported 12.73%, which compared against the *average* classical configuration) and the best static-embedding run by 6.7 points.
2. **Typical cost.** A transformer run trains 75× longer than a classical run and 32× longer than a static-embedding run (127× accelerator-time; geometric means of per-corpus medians).
3. **Task dependence.** The lead is 0.1–2.3 points on single-label corpora and 3.7–7.0 on multi-label corpora. On single-label corpora a linear run within 0.5–2.7 points costs 0.2–0.5% of the best run's time.
4. **Classical ML beats static-embedding NNs** at peak on 8/8 corpora, and no static-embedding run is on any Macro-F1 Pareto frontier.
5. **New: model size.** DeepSeek-R1-Distill has the best average rank but takes 6.6× BERT's time; Llama-3.2-1B ranks 6th of 7 at 5.1×. The large models versus the best base encoder: median −0.4 points at 4.9× the time.
6. **New: class embeddings.**
   - WCEs help static NNs (129 pairs, median +0.8, p = 1.5e-6).
   - In classical ML, WCEs help weak representations (dot +10.9, solo +3.5) but hurt `cat-doc` (−2.1).
   - **TCEs have no effect on transformers** (79 pairs, median 0.0, p = 0.66).

## Corrections to V6 confirmed by the run logs

- **TCE claim.** V6 said TCEs "actively degrade" transformer performance. The paired runs show no effect: 39 pairs up, 35 down, 5 tied. This was one of V6's listed contributions.
- **Hardware.**
  - Transformers ran with 4 GPUs on three machines (A100 40 GB, RTX A6000, RTX 6000 Ada), not only on RTX 6000 Ada; this explains V6's "A100" mention.
  - Static NNs ran on 1× RTX 6000 Ada for **all** corpora. V6's statement that the RCV1 neural runs used the Mac is wrong.
  - Classical ML ran mostly (78%) on the Linux workstation's CPUs. The RCV1 classical runs were on the M3 Max, and the DeepSeek-feature runs on Linux.
- **BERT checkpoint.** Fine-tuned runs used `bert-base-uncased` (30,522-token vocabulary in the log), not `bert-base-cased`. DistilBERT ran in both its cased and uncased variants.
- **Collapsed runs.** 9 of 528 neural runs score Macro-F1 below 0.35:
  - arXiv: 3 HF-SC runs at 0.000 and 5 LSTM runs;
  - IMDB: 1 Llama run that predicts a constant class.

  These distort V6's means.

Also carried over from the first V7 pass:

- GPT-3 → GPT-2;
- ELMo is a biLSTM;
- the truncated "alo" bullet;
- unit errors (".077%" for 7.7 points);
- fastText was run without subwords;
- DeepSeek and Llama descriptions and citations;
- dataset citations (arXiv, OHSUMED) and counts (BBC 1,490; RCV1 101 classes);
- BERT's input-embedding figure (V6 Fig. 4) credited to an unrelated paper;
- the model-table errors.

## Reference audit

All 89 V6 entries were checked against authoritative records; reports are in `refaudit/batch_{A,B,C,D}_audit.md`.

- **Results:** 14 entries were correct, 73 needed metadata fixes, and 19 were cited for claims they don't support.
- **Fabricated or wrong records:**
  - fabricated author lists: touvron2023llama, guo2025deepseek, qiu2020pre, minaee2021deep;
  - a broken DOI (zang2021research);
  - a wrong arXiv ID (leimeister2019skip);
  - wrong report numbers or venues.
- **V7 citations:** V7 cites 87 references, all audited or newly verified. `lc_v7_refs.bib` keeps 123 audited entries; only cited ones print.
- **Second pass (2026-10-23), claims against sources.** All 87 keys resolve, BibTeX reports no warnings and the printed list has no formatting faults. Each sentence that cites a source was re-read against the source; 11 were corrected:
  1. Energy-cost sentence (Intro): treviso2023efficient dropped. It surveys efficiency methods and is not a source for documented energy or dollar costs; patterson2021carbon and luccioni2024power carry the claim.
  2. galke2022bag (Intro): the wide MLP "matches or beats graph-based classifiers ... although fine-tuned transformers still lead" (was "rivals graph and sequence models").
  3. bucher2024fine, edwards2024language (Intro): fine-tuned small models beat *prompted* generative LLMs, "whether the LLMs are used zero-shot or with in-context examples" (was "zero-shot" only).
  4. trust2024study (Background): the study "revisit[s] text classification in the age of large language models" (was "examine how to fine-tune ... for classification").
  5. joulin-etal-2017-bag (Background): fastText "is often on par with deep classifiers in accuracy while training orders of magnitude faster" (was "trains in seconds at near state-of-the-art accuracy").
  6. galke2022bag (Background): "matches or outperforms graph-based classifiers, while fine-tuned BERT and DistilBERT remain ahead" (was "competitive with graph-based and fine-tuned models").
  7. moreo2021, WCE definition (Background): WCEs come from term–label co-occurrence in the training documents; the supervised functions (normalised dot product, PPMI, information gain, χ²) are attributed to the code base (`--supervised-method dotn|ppmi|ig|chi`), not to the paper.
  8. moreo2021, reported gains (Background): "consistent gains when WCEs are added to four neural architectures for topic classification" (was "for both linear and neural classifiers").
  9. li2020pytorch… (Framework): cited for DataParallel being single-process and less efficient than distributed data parallelism, which the paper establishes; the old wording ("scales sub-linearly") was not a claim the paper is a direct source for.
  10. dror2018hitchhiker (Framework): cited for test-selection guidance (non-parametric tests, exact *p*-values), not for a "no significance stars" recommendation.
  11. moreo2021, replication (Results): "consistent with the gains reported by [Moreo et al.], here across eight corpora" (was "replicates ... on a broader set of corpora"; the corpora and setup differ).

## Figures (all regenerated from run-level data)

1. Framework (TikZ).
2. Embedding layer, static vs. transformer (TikZ).
3. Peak gaps.
4. Critical-difference diagrams.
5. Pareto frontiers over every run.
6. Per-model results (new).
7. Pooled embeddings vs. best representation.
8. Paired class-embedding effects (new).

No chart uses a dual axis.

## Please confirm or decide

1. The Transformers version on the Ubuntu box (red). PyTorch 2.4.1 is confirmed by `bin/setup.sh`.
2. Whether the classical module's BERT features used the cased model (the code default) or the uncased one. The logs record the vocabulary only for fine-tuned runs.
3. The Acknowledgements say the authors reviewed and edited all AI-assisted material. Make sure that is true before submission.
4. Received/Accepted/Published dates, volume, pages and DOI are placeholders (`**`) for SCIRP to fill.
