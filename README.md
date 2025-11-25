# Design Notes

- Used LLM Guard (https://github.com/protectai/llm-guard) as a check against prompt injection or other undesirable inputs or outputs.
- Used open source/open weight models in order to find the most powerful and efficient small language model that can be run locally. 
-- For the base language model, 'allenai/OLMo-1B-hf' and "HuggingFaceTB/SmolLM2-1.7B-Instruct" were considered before settling on "HuggingFaceTB/SmolLM3-3B". The former was one of the higher ranked models in Hugging Face's AI Energy Score leaderboard (https://huggingface.co/spaces/AIEnergyScore/Leaderboard) for Text Generation.
-- For the document embedding model, "avsolatorio/GIST-all-MiniLM-L6-v2" was considered before settling on "sentence-transformers/all-MiniLM-L6-v2". The former was one of the higher ranked models in Hugging Face's AI Energy Score leaderboard (https://huggingface.co/spaces/AIEnergyScore/Leaderboard) for Sentence Similarity.
- Time permitting, more work could be done on standardizing tool-calling frameworks (ex. Model Context Protocol), performance evaluation/benchmarking or monitoring (AgentOps or LangSmith). Evaluation could be done using an LLM-as-a-judge and a synthetic test dataset relevant to the problem, or using a standardized benchmark dataset such as MemBench (https://aclanthology.org/2025.findings-acl.989/) or MemoryBench (https://arxiv.org/abs/2510.17281).
