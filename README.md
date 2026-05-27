# MLOps & Deployment — A From-Scratch Engineering Codex

A six-week, high-level curriculum that re-implements the core of the modern MLOps
and model-serving stack from first principles. Every abstraction — experiment
tracking, model registries, containers, inference servers, CI/CD pipelines, drift
detectors, and LLM serving engines — is **built in plain Python/NumPy before it is
used**, then mapped onto the industry-standard tool it corresponds to. By the time
you reach MLflow, Docker, GitHub Actions, or vLLM, there is no magic left.

This repository is the hands-on companion to the lecture track: each week is a
single, self-contained Jupyter notebook that derives the systems problem, builds a
minimal working implementation you fully own, and then connects it to production
reality.

## Philosophy

- **From scratch, then in practice.** Each section has three layers — 🧠 *Theory*
  (the systems problem and why it exists), 🔧 *From scratch* (a dependency-light
  implementation you build), and 🏭 *In practice* (how the real tool does the same
  thing, and what it adds).
- **CPU-only and self-contained.** Every notebook runs top-to-bottom on a laptop
  with just NumPy. No GPU, no cloud account, no paid services required.
- **One idea, reused everywhere.** Content-addressable hashing introduced in Week 0
  reappears as the backbone of registries, container layers, and pipeline caching.
  Each week builds on the last; together they climb the MLOps maturity ladder from
  "train in a notebook" to a self-healing, monitored, LLM-capable deployment stack.

## Curriculum

| Week | Notebook | Theme |
|------|----------|-------|
| 0 | `week0_foundations.ipynb` | The MLOps mindset, reproducibility, and the artifact graph. Build content hashing and the run manifest. |
| 1 | `week1_tracking_and_registry.ipynb` | Build a file-backed experiment tracker and a versioned model registry with stage transitions — your own mini-MLflow. |
| 2 | `week2_packaging_and_containers.ipynb` | A self-describing model package format; what a container *actually* is (namespaces, cgroups, content-hashed layers); a production Dockerfile. |
| 3 | `week3_serving_and_inference.ipynb` | An inference server from raw sockets to a router; dynamic batching; the latency/throughput frontier and SLO metrics. |
| 4 | `week4_cicd_and_testing.ipynb` | A cached pipeline DAG; the tests only ML needs (data validation, behavioral, champion/challenger); a GitHub Actions workflow. |
| 5 | `week5_monitoring_and_drift.ipynb` | Data vs. concept drift; PSI, the KS test, and a CUSUM change-point detector; three-layer observability; closing the loop to retraining. |
| 6 | `week6_llm_deployment.ipynb` | What makes LLMs uniquely hard to serve: the KV-cache, quantization, continuous batching, and the throughput/cost math of autoregressive generation. |

## What you build, week by week

- **Week 0** — `hash_array`/`hash_obj`, an environment-capture utility, deterministic
  seeding, and a `RunManifest` with a reproducibility key.
- **Week 1** — `TrackingStore` (params/metrics/artifacts logging + search) and
  `ModelRegistry` (versions, Staging/Production/Archived transitions, audit log,
  rollback).
- **Week 2** — the `.mlpkg` self-describing package format, dynamic loading from a
  manifest, model signatures, and an `ImageBuilder` that models Docker's layer
  caching.
- **Week 3** — an HTTP parser, an `InferenceApp` router with `/health` and
  `/metrics`, a `DynamicBatcher`, and a latency-percentile reporter.
- **Week 4** — a `Pipeline` DAG executor with content-based caching, behavioral
  (directional/invariance) tests, a `champion_challenger_gate`, and a fail-fast CI
  run mapped to a GitHub Actions YAML.
- **Week 5** — `psi`, `ks_2samp_statistic`, a `CUSUM` detector, a three-layer
  monitoring controller, and a drift-triggered retraining loop.
- **Week 6** — a KV-cache attention loop, `quantize_int8`, a continuous-batching
  simulator, and an LLM serving cost/capacity estimator.

## Getting started

```bash
# Clone, then create an environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Launch
jupyter lab     # or: jupyter notebook
```

Open the notebooks in order, starting with `week0_foundations.ipynb`, and run each
cell top to bottom. Every notebook is independently runnable.

## Prerequisites

Comfortable Python and basic NumPy. Having trained at least one model before is
enough — any of the courses in this profile (e.g. *Deep Learning*, *Generative
Artificial Intelligence*, *Reinforcement Learning*) provides ample background. The
LLM-specific Week 6 is most rewarding if you've seen transformer attention before;
the *Advanced LLM Architectures & Optimization* and *High-Performance AI
Infrastructure & LLMOps* courses pair naturally with it.

## Capstone (suggested)

Integrate all six weeks into one end-to-end system: a pipeline (W4) that trains a
model, logs it to your tracker and registry (W1), packages it (W2), serves it
behind your batching server (W3), monitors it for drift (W5), and — if the model is
an LLM — applies the KV-cache, quantization, and continuous batching from W6.
Wire the drift detector's alarm to re-trigger the pipeline so the loop closes
itself. The result is a model deployment that retrains, re-validates, and
re-promotes without a human in the loop — MLOps maturity level 4, built entirely
from parts you understand.

## A note on scope

These notebooks implement the *core idea* of each tool to make it legible, not a
production replacement for it. In real systems you should reach for MLflow, Docker,
Kubernetes, GitHub Actions, Evidently/NannyML, and vLLM/TGI — the goal of this
course is to ensure that when you do, you know exactly what each one is doing
underneath, and why.
