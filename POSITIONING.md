# Positioning Memo: MetabolonR-LLM

**Working title:** *MetabolonR-LLM: A Reproducibility-First LLM Agent for Metabolomics Pipeline Automation*

---

## 1. The Landscape: What Has Been Built

The past 18 months have produced a cluster of papers applying LLM agents to biological data analysis. Six are directly relevant to our positioning.

---

**CellAgent** (Xu et al., arXiv 2407.09811, 2024)
Built a hierarchical multi-agent system for single-cell RNA-seq analysis using GPT-4, with three specialised LLM roles — planner, executor, and evaluator — coordinated by a self-iterative optimisation loop. Achieved a 92% task completion rate, more than doubling the rate of direct GPT-4 usage. *What it missed:* The system is single-domain (scRNA-seq), tightly coupled to GPT-4, and provides no mechanism for running or auditing analyses without the LLM present. Reproducibility is not addressed.

---

**BioAgents** (Mathur et al., arXiv 2501.06314, 2025)
Built a three-agent system on Phi-3 small language models — two specialised agents (one fine-tuned on Biocontainers documentation, one using RAG on nf-core/EDAM ontology) plus a reasoning agent — designed to run locally without large compute. Achieved ROUGE-1 of 0.121 on the Biostars QA benchmark, near-identical to GPT-4o (0.122). *What it missed:* Human evaluation used only five experts. The system fails on medium-to-hard code generation tasks, reverting to step outlines. ROUGE-1 is a poor validity metric for analytical correctness. No pipeline execution layer.

---

**MetaboT** (Rutz et al., arXiv 2510.01724, 2025)
Built a modular multi-agent LLM framework that translates natural-language questions into executable SPARQL queries over the Experimental Natural Products Knowledge Graph (ENPKG), with specialised agents for scope validation, entity resolution, schema-aware query generation, and result interpretation. Validated on an expert-authored benchmark of NL–SPARQL pairs. *What it missed:* The system addresses knowledge retrieval, not quantitative analysis pipelines. It requires structured graph infrastructure to operate and produces query results, not processed datasets. No statistical analysis layer.

---

**MetaBench** (Zhou et al., arXiv 2510.14944, 2025)
Introduced the first dedicated benchmark for evaluating LLMs in metabolomics — 8,100 samples across five capability dimensions: knowledge, understanding, grounding, reasoning, and research. Found that cross-database identifier grounding is catastrophically difficult even for frontier models: without retrieval augmentation, the best-performing model achieves only 0.87% accuracy on grounding tasks. *What it missed:* MetaBench evaluates static LLM capabilities, not agentic pipeline execution. It does not test whether an agent can correctly run a differential abundance analysis or impute missing values. There is no tool-calling or execution layer.

---

**MSAgent** (preprint, bioRxiv 2026.04.22.720103)
Orchestrated a toolbox of over 50 domain-specific mass spectrometry tools via LLMs, with two traversal modes: depth-first Chat for single-spectrum iterative analysis and breadth-first Batch for large-scale parallel execution. *What it missed:* The system is mass spectrometry-specific and does not generalise to metabolomics quantitative data pipelines (QC filtering, imputation, differential abundance). Batch mode claims reproducibility but does not expose a session log or LLM-free replay mechanism.

---

**BixBench** (Nathansen et al., arXiv 2503.00096, 2025)
Not a system, but a benchmark: 50+ real-world biological data analysis scenarios with nearly 300 open-answer questions, specifically designed to measure long, multi-step LLM agent reasoning trajectories on biological datasets. *What it missed:* BixBench characterises the problem space rather than solving it. It does not propose an architecture. Current frontier agents perform poorly on its tasks, indicating that the field lacks validated solutions for multi-step biological analysis.

---

## 2. The Gap

Taken together, the prior work shares three structural limitations:

**No LLM-free reproducibility.** Every system reviewed requires the LLM to be present to re-run an analysis. There is no mechanism by which a computational biology colleague, a reviewer, or a journal could reproduce results without re-invoking the model with the same non-deterministic prompt.

**Loose tool coupling.** In most systems, "tools" are code snippets generated and executed by the LLM at runtime. The tool layer is not independently auditable or runnable. The LLM is inside the execution path, not above it.

**No metabolomics pipeline coverage.** None of the reviewed systems implement the standard quantitative metabolomics workflow: missingness-based QC filtering, KNN imputation, log transformation, autoscaling, PCA, differential abundance with multiple-testing correction, and sources of variation decomposition. MetaBench evaluates LLM knowledge of metabolomics concepts; it does not run metabolomics analyses.

---

## 3. Where MetabolonR-LLM Fits

MetabolonR-LLM is designed around a single architectural principle: **the LLM is a planner, not an executor.** All analytical work is performed by a deterministic, independently runnable tool layer. The LLM decides which tools to call and in what order; it never touches the data directly.

**Reproducibility-first design.** Every tool invocation — including the tool name, all input parameters, and a structured result summary — is written to a session log at runtime. The session log is a complete, ordered record of the analysis. A second researcher can replay the exact pipeline from the JSON file without access to the LLM, the original prompt, or any model at all. This is a design property, not a post-hoc export.

**Deterministic tool layer.** Each of the 11 tools in `tools/` is a pure Python function. Tools do not call the LLM. Tools do not generate code. A tool always produces the same output for the same input. This makes individual steps unit-testable and makes the full pipeline auditable by inspecting code, not model weights.

**Metabolomics-native pipeline.** MetabolonR-LLM implements the canonical MetabolonR workflow end-to-end, with each step exposed as a callable tool: `load_metabolomics_data` → `qc_filter` → `impute_missing` → `transform` → `scale` → `pca` → `sources_of_variation` → `differential_abundance`. Parameter choices (imputation neighbours, missingness threshold, p-value correction method) are explicit tool inputs, not implicit model decisions, and are therefore captured in the session log.

**Validation approach.** Because the tool layer runs independently of the LLM, we can validate the system in a way that prior work cannot: run the full pipeline with the LLM agent directing tool calls, then replay the logged session without the LLM and confirm identical outputs. This constitutes a reproducibility test, not a capability benchmark. We can further compare LLM-directed parameter choices against analyst-specified defaults as an internal consistency check.

---

## 4. Contribution Summary

| Property | CellAgent | BioAgents | MSAgent | MetaboT | MetabolonR-LLM |
|---|---|---|---|---|---|
| Metabolomics pipeline | — | — | Partial (MS) | Query only | Full |
| LLM-free replay | No | No | No | No | **Yes** |
| Deterministic tool layer | No | No | No | Partial | **Yes** |
| Session logging | No | No | No | No | **Yes** |
| Multi-testing correction | — | — | — | — | BH / Bonferroni |
| Validated by replay | No | No | No | No | **Yes** |

The core claim of the paper is not that an LLM can reason about metabolomics data — MetaBench has already shown the limits of that. It is that an LLM agent can correctly orchestrate a validated, reproducible metabolomics pipeline, and that correct orchestration can be verified independently of the model.

---

*References: arXiv:2407.09811 · arXiv:2501.06314 · arXiv:2503.00096 · arXiv:2510.01724 · arXiv:2510.14944 · bioRxiv 2026.04.22.720103*
