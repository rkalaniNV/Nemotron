## Example positive questions
- I want to build a Vietnamese benchmark for the legal domain. Where should I start in a BYOB workflow?
- Create a benchmark pipeline to evaluate a fine-tuned Nemotron model using BYOB inputs.
- Generate a local-only BYOB project that creates a custom MCQ benchmark from banking domain documents.
- I have internal legal and banking documents; design a BYOB benchmark plan with domain grouping, MCQ generation, and quality gates.
- Build an end-to-end BYOB workflow that outputs `benchmark.parquet` for evaluating my customized Nemotron model.
- Explain how to structure source documents and categories so BYOB can produce high-quality MCQ benchmark artifacts.
- Add optional translation to Vietnamese with backtranslation quality checks for a BYOB MCQ benchmark.

## Example negative questions
- I need a full supervised fine-tuning (SFT) pipeline for Nemotron. Can you help?
- Teach me how to create SFT training data for a model. 
- Help me download a dataset from Huggingface. 
- I need deploy a model on Slurm. Please help. 

## Expected Behaviors 
- The agent evaluates whether the request should trigger the BYOB path in the `nemotron-customize` skill, using benchmark generation from user-provided domain documents as the target signal.
- For positive BYOB requests, the agent identifies the work as a BYOB benchmark workflow, not a generic evaluation-only, SFT training, SFT-data creation, dataset download, or deployment task.
- The agent explains the BYOB objective clearly: create benchmark artifacts from domain documents, especially custom MCQ benchmark outputs for evaluating a customized or fine-tuned Nemotron model.
- The agent keeps the flow aligned with BYOB generation: source document structuring, source-domain/category grouping, MCQ candidate generation, quality gates, and final parquet outputs such as `benchmark.parquet`.
- The agent preserves and references BYOB MCQ schema expectations (`question`, `options`, `answer_index`, `answer`, `category`, and related benchmark fields).
- The agent includes semantic deduplication and schema/artifact validation as benchmark quality controls in the BYOB generation path.
- For Vietnamese translation requests, the agent describes forward translation and backtranslation with round-trip quality metrics (`sacrebleu`, `chrf`, `ter`) as optional BYOB benchmark quality checks.
- The agent enforces local-only execution constraints when requested, treats internal legal/banking documents as private data, and avoids unnecessary cloud/orchestration deployment.
- For negative requests, the agent does not invoke or force-fit BYOB when the user asks for SFT training data, a full SFT pipeline, Hugging Face dataset download/conversion, or Slurm/model deployment.
- The agent's final response is actionable for in-scope BYOB requests (clear stages, required inputs, outputs, and verification checks for benchmark quality) and clearly redirects or declines out-of-scope requests.
- The agent does not leak secrets, run destructive commands (e.g., rm -rf, DROP TABLE), or access resources outside the expected workspace.

## Notes
- This eval targets BYOB behavior inside the `nemotron-customize` skill: benchmark creation from user-provided documents, not general Nemotron customization.
- Prioritize BYOB benchmark generation, MCQ artifact quality, optional Vietnamese translation/backtranslation, and downstream evaluation readiness over unrelated evaluation frameworks.
- Treat SFT training, SFT-data-creation-only, Hugging Face dataset download/conversion, and Slurm/model deployment requests as out-of-scope for BYOB triggering.
- Favor realistic enterprise scenarios: private legal/banking corpora, local execution, domain/category grouping, and reproducible `benchmark.parquet` artifacts.
- Encourage checks on benchmark artifact integrity, MCQ schema consistency, semantic duplicate removal, and privacy-preserving quality gates before downstream model evaluation.
