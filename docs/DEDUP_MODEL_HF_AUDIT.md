# Dedup Model Hugging Face Audit

Дата аудита: 2026-06-27.

Цель: зафиксировать, как модели из research-ноутбуков и
`research/dedup/model_registry.py` ожидают input formatting, prompts,
pooling, normalization и runtime-настройки. Это не финальный выбор модели, а
памятка, чтобы не выбирать prefix/pooling на глаз.

## Короткий вывод

1. `intfloat/multilingual-e5-small` теперь используется у нас с
   `query: ` prefix. По HF card это правильнее для SKU-to-SKU symmetric
   similarity: E5 рекомендует `query: ` для symmetric similarity,
   paraphrase retrieval, bitext mining и embeddings-as-features.
2. `sergeyzh/BERTA` действительно mean-pooling sentence embedding model, но у
   нее много task prefixes. Текущий smoke-test с `paraphrase: `, no prefix и
   `search_document: ` полезен, но `categorize: ` и
   `categorize_entailment: ` тоже документированные кандидаты.
3. `deepvk/RuModernBERT-base` не sentence embedding model. Это raw Masked LM
   encoder; mean pooling в нашем smoke notebook — экспериментальная эвристика,
   а не рекомендованный embedding pipeline.
4. `Qwen/Qwen3-Embedding-4B` использует last-token pooling, а не mean pooling.
   Query side должен получать instruction; documents — без instruction.
5. `Qwen/Qwen3-Reranker-*` поддерживают custom instruction. Наш SKU prompt
   conceptually правильный, но сейчас текст prompt sauce-specific и должен
   стать category-neutral или category-run-specific.
6. Все reranker/cross-encoder модели по природе asymmetric query-document.
   Для SKU пары стоит отдельно проверить bidirectional scoring:
   `score(a,b)`, `score(b,a)`, `avg`, `max`.

## Scope

Фактические модели найдены в:

- `research/dedup/model_registry.py`;
- `notebooks/01_candidate_generation.ipynb`;
- `notebooks/03_matching_comparison.ipynb`;
- `notebooks/tmp_embedding_test_rumodernbert_berta_sauces.ipynb`.

HF-аудит покрывает:

- `intfloat/multilingual-e5-small`;
- `intfloat/multilingual-e5-base` как default в `BiEncoderConfig`;
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`;
- `cross-encoder/ms-marco-MiniLM-L6-v2`;
- `Qwen/Qwen3-Reranker-0.6B`;
- `Qwen/Qwen3-Reranker-4B`;
- `Qwen/Qwen3-Embedding-4B` как HF-аналог Polza id `qwen/qwen3-embedding-4b`;
- `BAAI/bge-reranker-v2-m3`;
- `jinaai/jina-reranker-v3`;
- `deepvk/RuModernBERT-base`;
- `sergeyzh/BERTA`.

`openai/text-embedding-3-small` и `openai/text-embedding-3-large` в registry
идут через Polza/OpenAI-compatible embeddings API, а не через Hugging Face
model card; для них нужен отдельный API/provider-аудит.

## Summary Table

| Model | Our current usage | HF-documented formatting/settings | Status |
| --- | --- | --- | --- |
| `intfloat/multilingual-e5-small` | `SentenceTransformer`, mean pooling + normalize from ST module; registry `text_prefix="query: "` | Inputs should start with `query: ` or `passage: `. For symmetric similarity/paraphrase retrieval/features, use `query: `. Mean pooling + Normalize module, max length 512. | Fixed for SKU symmetric retrieval. |
| `intfloat/multilingual-e5-base` | Code default in `BiEncoderConfig`, not notebook default; E5 fallback prefix now `query: ` | Same E5 rules as small; mean pooling + normalize, max length 512. | Fixed for SKU symmetric retrieval. |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `CrossEncoder.predict([(a,b)])`, raw score, no prefix | Query-passage text-ranking cross-encoder. Default activation is Identity; max length 512. | OK as raw rerank baseline, but asymmetric. |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Custom model example in reranker benchmark | Query-passage text-ranking cross-encoder. Default activation is Identity; max length 512. | OK as raw rerank baseline, but asymmetric and English/MS MARCO-biased. |
| `BAAI/bge-reranker-v2-m3` | `CrossEncoder`, raw score, batch 8 | Reranker gets query and passage, outputs relevance score; sigmoid/normalize optional. HF example uses `FlagReranker(..., use_fp16=True)` and `max_length=512` in raw Transformers example; tokenizer max length is 8192. | OK raw reranker; thresholds must match raw vs sigmoid choice. |
| `Qwen/Qwen3-Reranker-0.6B` | `CrossEncoder`, custom `sku_match` prompt, batch 1 | SentenceTransformers CrossEncoder has `default_prompt_name="query"` and prompt "Given a web search query..."; raw scores are logit differences; sigmoid optional; custom instruction recommended. | Custom prompt is justified; make it category-neutral/category-aware. |
| `Qwen/Qwen3-Reranker-4B` | Same as 0.6B, registry forces CPU | Same as 0.6B; 4B is heavier. README examples use max length 8192 for Transformers path. | Same prompt issue; CPU default is pragmatic for local Mac. |
| `Qwen/Qwen3-Embedding-4B` | Via Polza id `qwen/qwen3-embedding-4b`, not local HF | Instruction-aware embedding. ST config has `query` prompt and empty `document` prompt. Pooling is last-token, Normalize module, dimension up to 2560, custom dimensions 32-2560. | If used via local HF: do not mean-pool; use ST or last-token pooling. For Polza, verify provider prompt/dim support. |
| `jinaai/jina-reranker-v3` | `AutoModel(..., trust_remote_code=True, dtype="auto")`, `model.rerank(query, docs)` | HF card requires `trust_remote_code=True`; model is listwise multilingual reranker, can process up to 64 docs in one context; context 131K; returns `relevance_score`. | OK wrapper shape; asymmetric and chunk size should be tuned. |
| `deepvk/RuModernBERT-base` | Smoke notebook uses `AutoModel` + manual mean pooling | HF card is Masked LM/fill-mask, not sentence-transformers. Context 8192, hidden 768. Optional `patched-tokenizer` revision fixes several Russian lowercase token splits. Usage example is `AutoModelForMaskedLM`, not embeddings. | Treat as raw encoder experiment only; consider patched tokenizer and compare CLS/mean if kept. |
| `sergeyzh/BERTA` | Smoke notebook uses raw `AutoModel` + manual mean pooling and three prefixes: `paraphrase: `, no prefix, `search_document: ` | Sentence-transformers model with mean pooling + Normalize, max length 512. Default prompt is `Classification` -> `categorize_entailment: `. Documented prompts include `query`, `passage`, `STS`, `PairClassification`, `Clustering`, etc. | Current smoke is useful but incomplete; add `categorize: ` / `categorize_entailment: ` if expanding. |

## Model Notes

### intfloat/multilingual-e5-small and e5-base

Sources:

- <https://huggingface.co/intfloat/multilingual-e5-small>
- <https://huggingface.co/intfloat/multilingual-e5-base>
- <https://huggingface.co/intfloat/multilingual-e5-small/blob/main/1_Pooling/config.json>
- <https://huggingface.co/intfloat/multilingual-e5-small/blob/main/modules.json>

Important details:

- The model cards explicitly say every input should have `query: ` or
  `passage: `.
- Rules from card:
  - asymmetric retrieval: query side `query: `, document side `passage: `;
  - symmetric semantic similarity / paraphrase retrieval / bitext mining:
    `query: `;
  - embeddings as features for classification/clustering: `query: `.
- SentenceTransformers modules are Transformer -> Pooling -> Normalize.
- Pooling config: mean tokens, not CLS/max.
- Max length: 512.

Our current registry setting:

- `MODEL_REGISTRY["embedding_e5_small"].text_prefix` is `query: `.
- `MODEL_REGISTRY["bi_encoder_e5_small"].text_prefix` is `query: `.
- `model_text_prefix()` fallback returns `query: ` for unknown E5 ids.

For SKU candidate generation, the task is closer to symmetric
product-to-product retrieval than web query-to-document retrieval. The current
default is therefore `query: `. A later controlled test can still compare:

- `query: `;
- `passage: `;
- no prefix.

### sergeyzh/BERTA

Sources:

- <https://huggingface.co/sergeyzh/BERTA>
- <https://huggingface.co/sergeyzh/BERTA/blob/main/config_sentence_transformers.json>
- <https://huggingface.co/sergeyzh/BERTA/blob/main/1_Pooling/config.json>

Important details:

- BERTA is a sentence embedding model for Russian/English.
- It distills FRIDA into `sergeyzh/LaBSE-ru-turbo`.
- README says FRIDA's main CLS pooling mode was replaced by mean pooling.
- SentenceTransformers modules are Transformer -> Pooling -> Normalize.
- Pooling config: mean tokens.
- Context/max length: 512.
- Default prompt in `config_sentence_transformers.json` is
  `Classification`, which maps to `categorize_entailment: `.

Documented prompts:

| Prompt name / label | Prefix |
| --- | --- |
| `query` | `search_query: ` |
| `passage` | `search_document: ` |
| `RUParaPhraserSTS`, `STS22`, `STS` | `paraphrase: ` |
| `RuSTSBenchmarkSTS`, `MassiveIntentClassification`, `MultilabelClassification` | `categorize: ` |
| `Classification`, `PairClassification`, `TERRa` | `categorize_entailment: ` |
| clustering tasks | `categorize_topic: ` |
| sentiment/classification tasks | `categorize_sentiment: ` |

For SKU duplicate retrieval:

- `paraphrase: ` is plausible for same-product text similarity.
- no prefix is a useful baseline because README reports competitive STS/PI.
- `search_document: ` is plausible for document-document retrieval but is not
  directly symmetric.
- `categorize: ` is worth testing if we expand because README reports the best
  STS row among listed prefixes.
- `categorize_entailment: ` is worth testing because it is default and maps to
  PairClassification/Classification in config.

### deepvk/RuModernBERT-base

Sources:

- <https://huggingface.co/deepvk/RuModernBERT-base>
- <https://huggingface.co/deepvk/RuModernBERT-base/blob/main/config.json>

Important details:

- This is `ModernBertForMaskedLM`, pipeline `fill-mask`.
- It is not a sentence-transformers embedding model.
- Context length: 8192.
- Hidden size: 768.
- README usage is `AutoModelForMaskedLM`, optionally with
  `attn_implementation="flash_attention_2"`.
- README has `patched-tokenizer` revision. It fixes several common Russian
  lowercase letters being split into multiple subword tokens.

Our smoke notebook:

- Uses `AutoModel` and manual mean pooling. That is an experiment, not a
  documented sentence embedding recipe.
- If RuModernBERT remains in testing, mark results as raw-encoder baseline and
  consider comparing:
  - base tokenizer vs `revision="patched-tokenizer"`;
  - mean pooling vs CLS pooling;
  - no prefix only, unless we define our own task prefix experimentally.

### cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

Sources:

- <https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1>
- <https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/blob/main/config.json>

Important details:

- Text-ranking cross-encoder.
- Usage is `CrossEncoder(...).predict([("Query", "Paragraph")])`.
- Default activation in config: Identity, so scores are raw logits.
- Max length: 512.
- No documented prefix/prompt.

For SKU duplicate scoring:

- The model is query-passage asymmetric. We currently pass side A as query and
  side B as passage. For entity resolution, benchmark bidirectional score:
  `score(a,b)`, `score(b,a)`, `avg`, `max`.

### cross-encoder/ms-marco-MiniLM-L6-v2

Sources:

- <https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2>
- <https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/main/config.json>

Important details:

- Text-ranking cross-encoder trained for MS MARCO passage ranking.
- Usage is `CrossEncoder(...).predict([("Query", "Paragraph")])`.
- Default activation: Identity.
- Max length: 512.
- No documented prefix/prompt.

For SKU duplicate scoring:

- Same asymmetric caveat as mMARCO.
- More English/MS MARCO-biased than multilingual alternatives, so keep it as
  optional baseline, not primary candidate.

### BAAI/bge-reranker-v2-m3

Sources:

- <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- <https://huggingface.co/BAAI/bge-reranker-v2-m3/blob/main/config.json>

Important details:

- Multilingual reranker based on `bge-m3`.
- README distinguishes rerankers from embedding models: reranker takes query
  and document and directly outputs similarity.
- Scores can be mapped to `[0,1]` with sigmoid/`normalize=True`.
- README example recommends `FlagReranker(..., use_fp16=True)` for speed with a
  small quality tradeoff.
- Raw Transformers example uses `AutoModelForSequenceClassification`,
  tokenized query-passage pairs and `max_length=512`.
- Tokenizer config max length is 8192.

Our current usage:

- We use `sentence_transformers.CrossEncoder` style loading and raw scores.
- That is compatible with threshold calibration as long as thresholds are
  calibrated on raw scores.
- Do not mix raw-score thresholds with sigmoid-normalized scores.

### Qwen/Qwen3-Reranker-0.6B and Qwen/Qwen3-Reranker-4B

Sources:

- <https://huggingface.co/Qwen/Qwen3-Reranker-0.6B>
- <https://huggingface.co/Qwen/Qwen3-Reranker-4B>
- <https://huggingface.co/Qwen/Qwen3-Reranker-4B/blob/main/config_sentence_transformers.json>

Important details:

- SentenceTransformers CrossEncoder model.
- `config_sentence_transformers.json`:
  - `model_type`: `CrossEncoder`;
  - `activation_fn`: Identity;
  - `default_prompt_name`: `query`;
  - prompt `query`: "Given a web search query, retrieve relevant passages that
    answer the query".
- README says scores are raw logit differences by default.
- Sigmoid activation can convert scores to probabilities.
- README explicitly recommends custom instruct for specific scenarios/tasks and
  says no instruction on query side can reduce retrieval performance.
- README Transformers path uses yes/no token scoring and `max_length=8192`.

Our current usage:

- Registry passes custom prompt:
  `Decide whether two ecommerce sauce products are the same SKU. Pay attention
  to brand, flavor or purpose, unit weight, total weight, and pack count.`
- This is directionally right because Qwen reranker is instruction-aware.
- Problem: the instruction is sauce-specific. For `soap` and `coconut_oil`,
  change to category-neutral text or generate instruction from
  `DEDUP_CATEGORY_RUN`.

Recommended prompt direction:

```text
Decide whether two ecommerce products are the same base SKU. Pay attention to
brand, product line, flavor or scent or purpose, unit size, total size, and pack
count. Treat different pack counts as the same base product when the underlying
product is the same.
```

### Qwen/Qwen3-Embedding-4B

Sources:

- <https://huggingface.co/Qwen/Qwen3-Embedding-4B>
- <https://huggingface.co/Qwen/Qwen3-Embedding-4B/blob/main/config_sentence_transformers.json>
- <https://huggingface.co/Qwen/Qwen3-Embedding-4B/blob/main/1_Pooling/config.json>

Important details:

- Instruction-aware embedding model.
- Context length: 32k in README.
- Embedding dimension up to 2560.
- Supports user-defined output dimensions from 32 to 2560.
- SentenceTransformers config:
  - `query` prompt:
    `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`
  - `document` prompt: empty string.
- Pooling config: last token pooling, not mean pooling.
- Normalize module exists.
- README says queries benefit from prompt; documents do not need instruction.
- README says custom task-specific English instructions are recommended.

Our current usage:

- Registry only uses Qwen3 Embedding through Polza id
  `qwen/qwen3-embedding-4b`, not local HF loading.
- Polza adapter sends raw text to `/embeddings`.
- We need provider-specific confirmation whether Polza exposes prompt/instruction
  or dimension controls for Qwen. The HF local recipe cannot be assumed to be
  applied by Polza.

### jinaai/jina-reranker-v3

Sources:

- <https://huggingface.co/jinaai/jina-reranker-v3>
- <https://huggingface.co/jinaai/jina-reranker-v3/blob/main/config.json>

Important details:

- Requires `trust_remote_code=True`.
- HF usage is `AutoModel.from_pretrained(..., dtype="auto",
  trust_remote_code=True)`, then `model.rerank(query, documents)`.
- It is a listwise document reranker.
- It processes query and documents in one context and extracts contextual
  embeddings from the last token of each document.
- It can process up to 64 documents simultaneously within 131K context.
- Returns sorted dicts with `relevance_score`, `document`, `index`.
- No external prefix list is documented for the public `rerank()` API.

Our current usage:

- Wrapper groups pairs by side A as query and side B as documents, chunked by
  `documents_per_query`.
- Registry uses `documents_per_query=8`; config default is 16.
- Since the model is asymmetric/listwise, benchmark chunk size and
  bidirectional scoring if we use it for pair classification.

### OpenAI / Polza Embeddings

Registry entries:

- `openai/text-embedding-3-small`;
- `openai/text-embedding-3-large`;
- `qwen/qwen3-embedding-4b`.

These are not Hugging Face model cards in our local flow. Current adapter sends:

- `model`;
- `input`;
- `encoding_format`;
- optional `dimensions`;
- optional `user`.

Known gaps:

- Need Polza provider docs/catalog for model-specific prompt/instruction support.
- Need to confirm whether `qwen/qwen3-embedding-4b` supports dimensions and
  instruction through Polza.
- Do not infer HF local pooling/prompt behavior for Polza responses.

## Next Actions

Recommended next changes, in order:

1. Optional: add a tiny E5 prefix benchmark on sauces to measure the size of
   the already-applied default change:
   - `query: `;
   - `passage: `;
   - no prefix.
2. Make Qwen reranker instruction category-neutral/category-aware.
3. Add `categorize: ` and `categorize_entailment: ` to BERTA prefix smoke only
   if the three current variants are inconclusive.
4. If rerankers become serious candidates, add bidirectional scoring mode for
   cross-encoder/Qwen/BGE/Jina.
5. For Qwen3 Embedding through Polza, inspect live Polza model metadata before
   assuming HF prompt/dimension controls.
