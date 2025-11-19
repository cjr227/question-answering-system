# Design Notes

- Planned to use LLM Guard (https://github.com/protectai/llm-guard) as a check against prompt injection or other undesirable inputs or outputs. However, its dependencies were not fully compatible with the existing setup.
- Used open source/open weight models in order to find the most powerful and efficient small language model that can be run locally. 
-- For the base language model, 'allenai/OLMo-1B-hf' and "HuggingFaceTB/SmolLM2-1.7B-Instruct" were considered before settling on "HuggingFaceTB/SmolLM3-3B". The former was one of the higher ranked models in Hugging Face's AI Energy Score leaderboard (https://huggingface.co/spaces/AIEnergyScore/Leaderboard) for Text Generation.
-- For the document embedding model, "avsolatorio/GIST-all-MiniLM-L6-v2" was considered before settling on "sentence-transformers/all-MiniLM-L6-v2". The former was one of the higher ranked models in Hugging Face's AI Energy Score leaderboard (https://huggingface.co/spaces/AIEnergyScore/Leaderboard) for Sentence Similarity.
- Time permitting, more work could be done on a user interface (ex. Gradio), standardizing tool-calling frameworks (ex. Model Context Protocol), performance evaluation/benchmarking or monitoring (AgentOps or LangSmith). Evaluation could be done using an LLM-as-a-judge and a synthetic test dataset relevant to the problem, or using a standardized benchmark dataset such as RAGBench (https://arxiv.org/abs/2407.11005).

# Data Insights

There were 3,349 messages by 10 users, sent between November 8, 2024 and November 8, 2025, collectively submitting a median of 338 messages per user during that time frame.

- There are upwards of 50 messages concerned with payment/financial transaction issues.
- Five messages appear to be truncated/sent prematurely ("I finally", "I want to", "Please send", "Arrange a guided mountain", "I'm expecting delivery of").
