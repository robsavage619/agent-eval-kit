# NL-to-SQL Agent Example

Natural language to SQL over the public [Chinook database](https://github.com/lerocha/chinook-database) (a sample music store schema).

## Setup

```bash
# Download Chinook SQLite (one-time)
curl -L https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite \
     -o examples/nl_to_sql_agent/chinook.db
```

## Golden set structure

Each case has:
- `question` — natural language input
- `expected.contains_keyword` — SQL keyword the generated query must include (structural check)
- `expected.no_error` — query must execute without SQLite error

The `LLMJudge` is used as a secondary judge for semantic correctness of the result set.

## Run

```python
from agent_eval_kit import GoldenSet, run_evals
from agent_eval_kit.judges import LLMJudge
from examples.nl_to_sql_agent.agent import run_agent

golden = GoldenSet.from_jsonl("examples/nl_to_sql_agent/golden_set.jsonl")
judge = LLMJudge(criteria="The SQL query correctly answers the question using proper syntax.")
run = await run_evals(golden, run_agent, judge)
```
