# AI assistance and reuse

OpenAI Codex provided substantive assistance with interpreting the coursework,
designing features and model comparisons, writing training/inference code, running
experiments, testing, code review, bug fixes, and drafting and restructuring the
report. Subagents assisted with requirements review, experiments, dependency and
test review, and report organization. The Release 1.0 cleanup, English translation,
and report formatting were also AI-assisted.

The student must understand the implementation and conclusions before submission.
This disclosure does not claim that the student independently authored or checked
every line of the AI-generated work.

The course supplies the data, fixed representations, parser and evaluator.
GSM8K questions/references retain their MIT license; the supplied candidate and
embedding provenance for Qwen3-0.6B remains in `code/data/NOTICE.txt` and the data
manifest. No external training data, extra pretrained weights, external answer
repository, or model API calls are used by training or inference.

Code uses NumPy, SciPy, scikit-learn, CatBoost, joblib and threadpoolctl; exact
versions are pinned in `code/requirements.txt`. ReportLab generates the English
report PDF from `REPORT_EN.md` with embedded Bitstream Vera fonts bundled with
ReportLab. Custom TrueType fonts can be selected through `MP2_REPORT_FONT` and
`MP2_REPORT_BOLD_FONT`. Report generation is independent of inference.

The report explicitly discloses extensive validation selection, zero training
coverage of the two profit templates, historical test results, and the distinction
between development accuracy and held-out scoring. Release 1.0 preserves the
selected method; test scores did not determine the cleanup or parameter choices.
