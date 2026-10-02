# Evaluation

Retrieval baselines and scoring for this benchmark. Given a paper, the task
is to retrieve the prior papers that inspired it. `core_query` asks for
whole paper inspirations, and `subfield_query` asks for the inspirations of
one specific idea thread inside the paper.

## Setup

```bash
pip install -r requirements.txt
```

Set API keys as environment variables. `.env` at the repo root is loaded
automatically. The scripts pick the right key from the model id's prefix.

| Model prefix     | Env var             |
|-------------------|---------------------|
| bare id            | `OPENAI_API_KEY`    |
| `google/...`       | `GEMINI_API_KEY`    |
| `anthropic/...`    | `ANTHROPIC_API_KEY` |

## Data

Download the dataset from `ScholarCatalyst/ScholarCatalyst` on Hugging Face
and lay it out at `<BENCH_DIR>` (see the repo root README.md).

```
<BENCH_DIR>/
├── corpus.jsonl
├── queries.jsonl
└── rels/
    ├── core_query.jsonl
    └── subfield_query.jsonl
```

Then point `BENCH_DIR` at that directory, either in `.env`

```
BENCH_DIR=/path/to/<BENCH_DIR>
```

or per command with `--bench-dir /path/to/<BENCH_DIR>`.

### Loading interface

Everything here reads through `utils.py`. A new baseline or eval script
should use the same three functions instead of parsing the jsonl files
directly.

```python
from utils import load_corpus, load_queries, load_run

corpus = load_corpus(bench_dir)
# {doc_id: {"id", "title", "text", "published", ...}}

queries = load_queries(bench_dir, "core_query")  # or "subfield_query"
# [{"id", "query_id", "question", "paper_id", "paper_published", "positive_docs", ...}, ...]
# positive_docs is joined in from rels/{query_type}.jsonl.
# queries.jsonl does not store it directly.

run = load_run(bench_dir, "bm25", "core_query")
# {query_id: [{"doc_id": ..., "score": ...}, ...]}, reads runs/<category>/<model>/<query_type>.jsonl
```

## Embedding models (dense baselines only)

`bm25` needs nothing extra. `gemini-2` and `text-embedding-3-large` are APIs
and only need the matching key from the table in Setup. If your baseline
needs one of the local dense embedding models instead (`bge-large`,
`qwen3-4b`, `qwen3-8b` (our main dense baseline), `openscholar`, `scincl`,
`specter2`), download it first.

```bash
python download_retrieval_models.py --models qwen3-8b   # or: --models all
```

Downloads into `retrieval_baseline/` at the repo root. Skips models already
present there.

## Running a baseline

```bash
python encode_corpus.py --model bm25 --bench-dir $BENCH_DIR
python retrieve.py      --model bm25 --bench-dir $BENCH_DIR
python evaluate.py      --model bm25 --bench-dir $BENCH_DIR
```

Or run several registered baselines end to end with the command below. It
encodes, retrieves and evaluates, and it skips steps whose output already
exists.

```bash
bash run_baselines.sh $BENCH_DIR
```

## LLM based baselines

Query rewriting and reranking around a fixed sparse or dense retriever are
separate scripts, since they call an LLM and cost money per run. Both print
a cost and time estimate and ask for confirmation before making any calls.

```bash
python llm_query_augment.py --pipeline single_hop --retriever bm25 --model gpt-5.4 --bench-dir $BENCH_DIR
python llm_rerank.py        --pipeline rerank      --retriever bm25 --model gpt-5.4 --bench-dir $BENCH_DIR
```

`llm_query_augment.py --pipeline` is `single_hop` (rewrite the question once),
`multi_query` (generate up to 5 augmented questions, fused with reciprocal
rank fusion), or `hyde` (generate hypothetical prior work abstracts and
search with those instead of the question).

`llm_rerank.py --pipeline` is `rerank` (one listwise call over the top
candidates), `rerank_tournament` (batched elimination rounds for a larger
pool), or `oracle_tournament` (injects the gold documents into the pool
first, an upper bound on the reranker alone, not a real baseline).

Both write to `runs/agentic/{descriptor}/{query_type}.jsonl` and print the
exact `evaluate.py` command to score the run afterward.

Every `--model` here takes any id directly. A bare id goes to OpenAI,
`google/<id>` goes to Gemini, `anthropic/<id>` goes to Claude, each through
its own API key, so trying a different backbone is just a flag change.

## Agentic baselines

`agentic_search.py` runs baselines where an agent searches the corpus itself
(a bash and ripgrep harness or a tool calling loop) rather than querying a
prebuilt index like `llm_query_augment.py` and `llm_rerank.py` do. The shared
pieces live in `agentic/`.

- `agentic/views.py` builds per query corpus views for agents that grep files.
  The corpus is split once into month buckets (`<bench>/agentic/corpus_by_month/`),
  and each query gets a directory of hard links to the buckets it may see, minus
  the query paper. Same date rule as `retrieve.temporal_filter`.
- `agentic/ranking.py` turns the agent's answer into a depth 100 ranking. It holds the
  ids the agent listed, then the ids it read during the run, then a date filtered
  retriever ranking of the original question (`--backfill`, default bm25).
  Every item records its segment, so recall@5 and recall@20 reflect the agent's own
  list while recall@100 is agent plus backfill.
- `agentic/trajectory.py` writes one JSONL per query with every tool call, LLM
  call, token count and cost, under `runs/agentic/<pipeline>/trajectories/`.
- `agentic/requests_log.py` writes `<qid>.requests.jsonl` next to each
  trajectory, one line per model request (tokens, list and metered cost, agent
  role, error), with the raw request and response under `<qid>.bodies/`.
- `agentic/runner.py` drives one agent over one query type, resumes finished
  queries, and keeps going when an agent raises (that query is logged to
  `<query_type>.failed.jsonl` and rerun on resume).

An agent is a module in `agentic/agents/` with
`run(query, view_dir, traj, tools) -> answer text`. Register it in
`agentic_search.py`'s `AGENTS`. The shipped agents are listed below.

- `stub` is a smoke agent with no API calls (it lists the retriever's top hits).
- `grep` has one `bash(command)` tool run inside the query's corpus view, in the
  style of `DCI-Agent-Lite` (ripgrep over month bucket files, at most 20 commands).
  Needs `--views` (enabled automatically).
- `toolcall` is an agent in the style of PaperScout. The model gets `search(query)`
  over the date filtered retriever and, when `<bench>/citations.jsonl` exists
  (`{"id", "cites": [...]}` per line), `expand(doc_id)`, with at most 5 tool rounds.
  Every pooled paper is judged once with a usefulness prompt, and the pool is
  returned best first. The backbone is set with `--model` (default `gpt-4.1`).
  `--full-text corpus_full_text.jsonl` adds `read(doc_id[, section])`, and `--read-policy`
  decides what it may show (the body is cut at 12k characters).
  - `full` shows the whole paper, plus the bibliography block after the cut.
  - `abs` shows the Abstract only.
  - `abs+intro` shows the Abstract plus the Introduction.
  - `abs+method` shows the Abstract plus the sections whose title mentions a method word.
  - `abs+refs` shows the Abstract plus the bibliography (`## References (N entries)`, at most 80 entries or 6k characters).
  - `abs+intro+refs` shows the Abstract, the Introduction and the bibliography.
  - `choose` adds `sections(doc_id)`, and the agent reads the sections it picks.
- `deepresearch` uses the same `search` and `read` with a different procedure. It makes a plan
  of sub questions, runs a seed search of the question as written, then makes up to 20 searches
  and 10 reads with a reflection every 5 tool calls. The first 3 searches are required through
  `tool_choice`, and a prose reply before that is nudged back to search at most twice. The pool
  holds up to 150 papers, and one synthesis call orders it (there is no per paper judge).
  `--read-policy` defaults to `abs+intro+refs` for it.

Claude models run through `anthropic_native.py`, an adapter that lets an
agent written against the OpenAI chat.completions interface run on
Anthropic's own Messages API instead, with prompt caching and thinking block
replay across turns.

Run and score a baseline with the commands below.

```bash
python agentic_search.py --agent stub --backfill bm25 --limit 5 --bench-dir $BENCH_DIR
python evaluate.py --model agentic/stub-bm25 --bench-dir $BENCH_DIR
```

`evaluate.py` reports `precision@k`, `recall@k` and `ndcg@k` for
`k = 5, 10, 15, 20, 25, 100`. For runs that record the ids the agent
touched, it also reports `trajectory_recall`, the share of positives seen anywhere in the run.
Any run that writes `runs/agentic/{pipeline}/{query_type}.jsonl` lines of the form
`{"query_id": "...", "ranking": [{"doc_id": "..."}, ...]}` is scored the same way, and
`evaluate.py` discovers every `runs/agentic/**` leaf directory automatically.
