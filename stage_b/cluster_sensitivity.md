# Sealed TriviaQA stem-cluster sensitivity

> Descriptive and non-gating. This does not alter the sealed 18/20 verdict.

| Model | Full CI low | Full > .50 | Geometric CI low | Geometric > .50 |
|---|---:|:---:|---:|:---:|
| Llama-3.1-8B-Instruct-4bit | 0.6983 | yes | 0.7187 | yes |
| Llama-3.2-3B-Instruct-4bit | 0.7984 | yes | 0.7984 | yes |
| Mistral-7B-Instruct-v0.3-4bit | 0.9205 | yes | 0.9175 | yes |
| Mistral-Nemo-Instruct-2407-4bit | 0.9408 | yes | 0.9408 | yes |
| Phi-3.5-mini-instruct-4bit | 0.7298 | yes | 0.7298 | yes |
| Phi-4-mini-instruct-4bit | 0.8294 | yes | 0.8294 | yes |
| Qwen2.5-7B-Instruct-4bit | 0.7925 | yes | 0.7929 | yes |
| Qwen3-1.7B-4bit | 0.7844 | yes | 0.5830 | yes |
| Qwen3-8B-4bit | 0.8886 | yes | 0.8953 | yes |
| gemma-3-4b-it-4bit | 0.7791 | yes | 0.7461 | yes |

Exchangeable unit: TriviaQA question stem. Candidate selection and OOB scoring otherwise follow the sealed procedure.
