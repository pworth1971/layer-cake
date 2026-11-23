# V9 revision notes (2026-11-22)

Draft: `lc_v9_11222026.tex` → `lc_v9_11222026.pdf` (32 pp., as V8).
Source of record: `../V8/lc_v8_10292026.tex`, as submitted on 2026-10-29. This revision responds to the 11 comments in SCIRP's typeset proof (`../-1680506.pdf`, received 2026-11-22).
The V8 notes (`../V8/REVISION_NOTES.md`) describe the response to the editor, and the V7 notes the data, journal format and reference audit. Only the changes are listed here.

## Build

Unchanged from V8, except for the file names:

```bash
cd analysis
python extract_tables.py && python extract_runs.py && python analyze.py && python tables.py && python figures.py
cd .. && latexmk -pdf lc_v9_11222026.tex
```

The build has no overfull or underfull boxes, no undefined references and no BibTeX warnings. The data, analysis scripts, tables and figures are copied from V8 unchanged.

## Response to the proof comments

- **Title footnote** ("A revised and extended version of [1] ..."): deleted, as the typesetter proposed. The Introduction's "Relation to the earlier version" paragraph carries the same statement.
- **How to cite:** the name form "Worth Jr., P.J." is confirmed (the 2025 article was printed as "Worth Jr., P."). Year, volume and DOI stem now follow the proof (2027, 17(1)).
- **Appendix cross-references:** the proof printed "Appendix 7.1/7.2/7.3". All five now cite the tables directly (Tables A1–A3), which also answers the request to cite Tables A1–A3 in the main text. §4.1 gains "(Table A1)" after the Micro-F1 leads.
- **Ref. [37] (Firth 1957):** the volume is cited without an editor. The entry now reads "Studies in Linguistic Analysis (Special Volume of the Philological Society)". This follows the usual citation of the volume; the title page itself was not checked.

## Other changes

- **Licence:** the V8 source said CC BY-NC 4.0. The signed copyright form (`ijis/copyright_signed-10282026.pdf`) selects CC BY, which is what the proof prints, so the override is removed and the class default (CC BY 4.0) applies.
- **Data and Code Availability** now points to `paper/scirp/V9`.
- **Acknowledgements:** the AI-use statement now also covers the proof corrections (Claude Fable 5.2, 22 November 2026). **The author should confirm this wording.**

## Typesetting errors found in the proof

These need no source change. They are in the corrections list sent to SCIRP with this revision.

- Table A1: the ArXiv-10 transformer time is printed as 45,048; the source and Table 5 have 42,048. All other generated table rows (224 in total) match the proof.
- Table A2: the second sub-heading is printed as "Macro-F1" instead of "Micro-F1".
- Table 10 and the Figure 8 caption: the subscript of $\Pi_{M+\mathrm{WCE}}$ is lost, and the caption has "withWCEs".
- Figure 1 caption: "layer cake" for "Layer Cake". Finding 3: "which requires" for "which require".
- Reference list: capitalisation or missing details in twelve references, and the DOIs of [16] and [75] dropped.

## Open

- The proof omits the "Author Contributions" section, which is still in the source. The author should decide whether to ask for it back.
- No new submission zip.
