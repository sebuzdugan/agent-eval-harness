# agent-eval-harness

A tiny, reproducible harness that shows **why a trustworthy agent evaluation needs
per-task isolation**, run live on [Tensorlake](https://tensorlake.ai) microVM sandboxes.

The benchmark debate is stuck on scores. The harder, under-written problem is the
**harness**: if the agent runs in the same environment the evaluator trusts, a
zero-capability agent can score high by tampering with the test machinery or just
claiming success. The fix is a causal chain:

> verification must be **hidden** from the agent → hiding it needs **per-task isolation**
> → cheap isolation needs **fast forks** → that is **Tensorlake + Harbor**.

Every number below comes from running this harness on real Firecracker sandboxes.

## Results (reproduced across runs)

| Experiment | Result | What it shows |
|---|---|---|
| **EXP0** determinism | same task x5 -> **5/5 identical** PASS | a bit-identical snapshot removes environment-induced variance |
| **EXP1** cheats vs verdicts | a *trusting* harness passes cheating agents **4/4**; ground-truth on visible checks **1/4** | trusting the agent is worthless; artifact checks help but are not enough |
| **EXP1b** held-out + isolation | honest passes held-out; `hardcode_visible` **fails** held-out | teaching-to-the-test is only caught by held-out checks the agent never saw, which stay secret only because each task runs in its own fork |
| **EXP2** lie rate | trusting **87%** vs ground-truth **33%** -> **53% lie rate**, +53pt inflation | naive scores massively overstate real capability |
| **EXP3** contamination | shared harness **3** spurious failures vs isolated **0** | one poisoned task silently corrupts every later task in a shared box; fork-per-task eliminates it |
| **EXP3** throughput | **~16 tasks/min at concurrency=1** | scales ~linearly as paid tiers lift the cap to 1,000+ |

The "cheats" are the ones documented in the literature: the SWE-bench `conftest.py`
hook that forces every test to report passed, editing the test file, and simply
printing "all tests passed". The held-out case hardcodes the visible tests and is only
caught by checks it was never shown.

## Run it

Requires Python >= 3.10 and a Tensorlake API key (the free tier works; it is capped at
1 concurrent sandbox, so the harness runs strictly sequential).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export TENSORLAKE_API_KEY=tl_apiKey_...      # from https://cloud.tensorlake.ai -> API Keys
python eval_harness.py
```

It builds one canonical task environment, snapshots it once, and forks a clean microVM
per task for every experiment. Get a free key and credits at the
[Tensorlake playground](https://tensorlake.ai).

## How it works

- **The task (`s0`)**: a buggy Python module plus a visible test file. Snapshotted once;
  every rollout is a fresh fork of that snapshot.
- **Scripted "agents"**: deterministic bash behaviors (honest fix, no-op, the three
  cheats, a partial fix, and a hardcode-to-the-visible-test). No model or model API key
  needed, so results are fully reproducible.
- **Trusting harness**: believes the agent's own `pytest` exit code or its stdout claim.
- **Ground-truth verifier**: imports the artifact directly (not via pytest, so a hijacked
  `conftest` cannot forge it) and checks behavior.
- **Held-out verifier**: the same, with checks the agent never saw, injected only at
  verify time into a fork the agent could not reach.

## Honest limitations

Verification is only as good as the checks you write; a finite oracle can be memorized,
which is exactly why held-out checks plus isolation matter. External services the sandbox
reaches over the network are outside the snapshot. There is a per-restore floor, and
snapshots consume storage.

## Background

Companion to the Medium piece *Your Agent Benchmark Is Lying to You* (link added on
publish). Primary sources the results line up with: the UC Berkeley RDI benchmark audit
(BenchJack), OpenAI retiring SWE-bench Verified (Feb 2026), Cursor on reward hacking,
METR on frontier reward hacking, Terminal-Bench / Harbor, and the run-to-run variance
study on agentic evals.

Built with [Tensorlake](https://tensorlake.ai) sandboxes and the
[Harbor](https://github.com/harbor-framework/harbor) evaluation framework.
