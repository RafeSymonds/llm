# Build Plan — Two Libraries That Feed Each Other

You're not reproducing tutorials. You're shipping two Python libraries you own, and each milestone below is specified by a **goal and a number**, not a video to copy. Videos and papers appear only as *reference you consult while building toward your own spec* — never as the thing you reproduce.

## The two artifacts

- **`ember`** — your GPT model + training library. A from-scratch, hackable LM codebase: pretraining, finetuning, evals, and the models you train with it. (Rename to taste.)
- **`forge`** — your GPU kernel-generation library. Starts as a compile-verify-profile harness, grows into an autotuner, then an agentic kernel generator, then one driven by a model *you finetuned*. (Rename to taste.)

## The north star — the loop (honest version)

```
ember (your finetuning stack + data)  ──trains──▶  forge's kernel generator
      ▲                                                  │
      │                                                  ▼
      └────accelerates the daily experiment loop──── custom fused kernels
```

Two honesty clauses, decided now so they don't ambush you later:

1. **"Your model" means your *lineage*, not your pretrained weights — at first.** A from-scratch 124M model pretrained on generic text cannot write compiling kernels (KernelLLM, the existence proof for small kernel models, is an 8B code-pretrained Llama). The B4 generator is a small open-weights code model finetuned with *your* `ember/finetune.py`, on *your* harvested data, against *your* evals. A kernel generator built on weights you pretrained from scratch is the long-horizon stretch goal, unlocked if/when ember scales past ~1B with code-heavy data. The loop still closes — through your library, not your pretraining run.
2. **The kernel speedups land where you train daily: the 4070.** Kernels are written and tuned on Ada (sm_89); the one-off GPT-2 run rents Hopper (sm_90) — different shared-memory sizes, tensor-core paths, and features. Tunings don't transfer, so don't pretend they do. `forge`'s wins are measured as tokens/sec on the `ember` A4 ablation loop, which runs constantly on your desk. The rented H100 run uses stock fast paths (`torch.compile`, SDPA) and is never gated on homegrown kernels.

The loop never "finishes" — it's the thing you improve indefinitely. Everything below builds toward closing it.

---

## How every milestone is specified

- **Goal** — one sentence, measurable.
- **Ship** — the API/module/artifact that exists at the end.
- **Metric** — the number that says you succeeded. The metric is the exit gate: you don't start the next milestone until it's hit or you've written down exactly why you're re-scoping. If there's no number, it's not a milestone.
- **Forced learning** — the concepts you *cannot avoid* understanding to hit the metric (this is where the education hides).
- **Learn** — the prime → build → deepen sequence (see [The learning system](#the-learning-system)): what to skim *before*, consult *when stuck*, and read properly *after* the metric is hit.
- **Trap** — the specific failure mode that will bite you here.
- **Time** — estimate at ~10 focused hrs/week. Scale linearly to your actual cadence.

---

# LIBRARY A — `ember`: your GPT model

## A0 — The correctness spine
**Goal:** train a character/byte-level LM whose held-out loss matches a known-good baseline on the same data budget.
**Ship:** `ember/model.py` (your Transformer: MHA, MLP, residual+norm, positional), `ember/train.py` (loop, AdamW, LR schedule, grad clip, checkpointing), `ember/data.py`, and `tests/` — starting with the overfit-one-batch test.
**Metric:** (gate test first) overfit a single batch to ≤0.01 loss — if it can't, the model is wrong. Then: held-out bits-per-byte within 2% of a nanoGPT baseline on the same dataset and step budget.
**Forced learning:** attention math, autograd behavior, why LR warmup/decay matters, gradient clipping, mixed precision.
**Learn:** *Prime:* watch Karpathy "Let's build GPT" once, end-to-end, **no code-along**. *Consult:* rewatch specific segments when stuck; diff against nanoGPT source only after your version runs. *Deepen:* Vaswani 2017 (*Attention Is All You Need*) after the metric is hit.
**Trap:** silent shape/broadcasting bugs that still "train" but underperform. The overfit test is written first, before the training loop is even finished.
**Time:** ~3 weeks.

## A1 — Your tokenizer and data pipeline
**Goal:** replace the toy tokenizer with your own BPE and a streaming data loader that keeps the GPU fed.
**Ship:** `ember/tokenizer.py` (train/encode/decode), a sharded/streamed memory-mapped dataset loader, and a vocab-size ablation writeup (loss + throughput at 3 vocab sizes) — the ablation is a deliverable, not a metric.
**Metric:** (1) round-trip on a ≥100MB corpus with **zero** mismatches; (2) training tokens/sec ≥95% of the identical run fed pre-tokenized in-memory synthetic batches — i.e., within 5% of the loader-is-free upper bound. (Not `nvidia-smi` "GPU utilization" — that reads high whenever any kernel is resident and tells you nothing about stalls.)
**Forced learning:** BPE, the tokenization→compute tradeoff, data-loader throughput, memory-mapped datasets.
**Learn:** *Prime:* Karpathy "Let's build the GPT Tokenizer," once, no code-along. *Deepen:* Sennrich 2016 (BPE).
**Trap:** tokenizer/data bugs are invisible in the loss curve until much later. The round-trip test runs in CI before anything real trains.
**Time:** ~3 weeks.

## A2 — Reproduce GPT-2 (124M) — your codebase, their numbers
**Goal:** hit GPT-2-class eval numbers with *your* `ember` codebase, not a fork.
**Ship:** a config + run that scales `ember` to 124M params; `ember/eval.py` with val loss + HellaSwag.
**Metric:** on FineWeb-Edu with a 10B-token budget: **val loss ≤ ~3.28 and HellaSwag ≥ 29.4%** (GPT-2 124M's score; llm.c's reproduction hit ~29.9% — that's the class you're matching). **This is the north-star model milestone.**
**Run configuration — decided now:** the run uses stock fast paths (`torch.compile`, PyTorch SDPA — which *is* FlashAttention). Your `forge` kernels do **not** gate this run; they prove themselves on the 4070 (see Convergence).
**Pre-flight checklist (all required before renting the 8×):**
1. Loop correct + overfit-tiny on the 2080S.
2. Tokenizer and eval set frozen — hashes recorded.
3. A 1-hour single-H100 dry run (~$3): validates throughput projection, logging, and checkpoint/resume — tested by killing the job on purpose and resuming.
4. A written throughput projection: tokens/sec × budget = wall-clock and cost, so a stall is detectable in the first 10 minutes.
**Forced learning:** distributed/large-batch training, gradient accumulation, compute-optimal scaling, eval harness design.
**Learn:** *Prime:* Karpathy "Let's reproduce GPT-2" (once, for the shape of the run). *Consult:* llm.c/nanoGPT configs when your numbers diverge. *Deepen:* Radford 2019 (GPT-2); Hoffmann 2022 (Chinchilla).
**Budget:** a clean llm.c-class run is ~90 min on 8×H100 (≈$50–100 at spot prices). Your first-time codebase will be slower and something will fail: **budget $150–300 for 2–3 attempts.** Never debug on the 8× node — reproduce on a single GPU.
**Trap:** eval leakage and mismatched tokenization make your numbers look better than they are. Freezing (checklist item 2) is the defense.
**Time:** ~4 weeks (mostly prep; the run itself is hours).

## A3 — The instruction/preference layer
**Goal:** build the finetuning stack, and prove it where the signal is measurable.
**Ship:** `ember/finetune.py` supporting full-FT, LoRA, and DPO (DPO before GRPO: offline, no reward model, no rollout infra — GRPO belongs to the kernel-RL stretch after B4). Proven on **two tracks**:
- *Track 1 — ember-124M:* SFT your own base model. Expect a small, largely qualitative lift — a 124M model is below the scale where DPO deltas separate from noise. Document honestly; this track is about the code path working on your lineage.
- *Track 2 — a small open-weights model (1–2B, Llama-3.2-1B / Qwen-class):* where instruction lift is real and measurable. LoRA fits on the 4070 (12GB); QLoRA if you go bigger. This track is also deliberate practice for B4, which uses the same machinery.
**Metric:** on Track 2, with a **pre-registered judge** (fixed judge model, fixed prompt, frozen 100-prompt eval set, position-swapped pairs): finetuned wins ≥65% vs its own base. Plus a LoRA-vs-full table (params, VRAM, wall-clock, win-rate) and a base-capability eval before/after — the alignment-tax check is mandatory, not optional.
**Forced learning:** SFT, RLHF vs DPO vs GRPO, LoRA/QLoRA mechanics and their tradeoffs, judge-eval design and its failure modes.
**Learn:** *Deepen:* Ouyang 2022 (InstructGPT); Rafailov 2023 (DPO); Hu 2021 (LoRA); Dettmers 2023 (QLoRA). *Consult:* `karpathy/nanochat` as the full-lifecycle reference.
**Trap:** preference training silently degrades base capability. Always eval the base task too, not just the preference win-rate.
**Time:** ~4 weeks.

## A4+ — The improvement engine (this is the "keep improving" part)
**Goal:** every architectural idea becomes a measured experiment, not a vibe.
**Ship:** `ember/experiments/` — each experiment is a PR containing a **pre-registered** `EXPERIMENT.md` (hypothesis, metric, compute budget, decision rule — written *before* the run) plus the result. Candidates: RoPE, RMSNorm, SwiGLU, GQA, QK-norm, better init/muP, data-quality filtering, longer context.
**Metric:** `ember/experiments/LEADERBOARD.md` — loss at fixed token budget on a standard 4070 config. Each merged change must move it or it gets reverted.
**Forced learning:** the modern LLM architecture delta from GPT-2 → today, and honest experimental methodology.
**Learn:** *Consult:* the Llama/Mistral/Qwen papers for the specific techniques; ablate them yourself rather than trusting the abstract.
**Trap:** stacking five changes at once so you can't attribute the win. One variable per experiment — the pre-registration template enforces it.
**Time:** ongoing; first three experiments ~1 week each.

---

# LIBRARY B — `forge`: your kernel-generation library

The order matters: you can't generate good kernels until you've built the harness that *verifies and profiles* them. So the deterministic core comes first — and it's also where you learn kernels by hand.

## B0 — The harness (the foundation everything stands on)
**Goal:** a library that takes a kernel (as source), JIT-compiles it, verifies correctness against a PyTorch reference, and reports speedup — safely and un-gameably.
**Ship:** `forge.run(kernel_src, reference_fn, inputs)` → `{correct: bool, speedup: float, profile: {...}}`, with:
- JIT compile paths for CUDA (torch `load_inline`) and Triton.
- Correctness on **fresh randomized inputs each run**, multiple shapes including non-power-of-2, with explicit tolerances.
- Honest timing: warmup iterations, CUDA events, cache-clearing between iterations, and a CUDA-events-vs-wall-clock cross-check to detect timer games.
- Sandbox: kernel code runs in a subprocess as a non-root user (container if convenient) with timeout and memory caps, no network. A wedged GPU is still possible; a wrecked machine isn't.
- Anti-cheat: verify the reference actually executed; check outputs aren't aliased or pre-computed (the Sakana memory-reuse exploit); re-verify any "pass" on shapes the kernel hasn't seen.
**Metric:** two gates. (1) Reproduce KernelBench's `fast_p` scoring within noise on ≥5 Level-1 problems using hand-written kernels you supply. (2) **Red-team your own harness:** write ≥3 deliberately cheating kernels (timer manipulation, output caching, degenerate-input exploitation) and confirm the harness catches all of them. This is your AI-security instinct as a deliverable.
**Forced learning:** CUDA JIT/loading, correctness tolerances, honest GPU timing, why naive timing lies.
**Learn:** *Prime:* GPU MODE lectures 1–3; skim the KernelBench paper's harness section. *Deepen:* the Sakana post-mortem — read it as a defender.
**Trap:** reward hacking. Sakana's kernel agent faked 10–100× by exploiting the eval harness. The anti-cheat is built *now*, while the only adversary is you — B3 is where it earns its keep.
**Time:** ~3 weeks.

## B1 — Hand-written kernels (learn the metal, populate the baseline)
**Goal:** beat PyTorch eager on a set of ops with kernels you wrote by hand.
**Ship:** `forge/kernels/` with tiled matmul, fused softmax, a fused elementwise chain, layernorm, RMSNorm, and one reduction/transpose — each registered and benchmarked by B0.
**Metric:** `fast_1` (correct AND >1× vs eager) on ≥6 ops on the 4070, each with an Nsight profile writeup explaining the win. **Also report every op against `torch.compile`** — that's the honest baseline; beating eager on fused elementwise is nearly automatic. Stretch: beat compile on ≥2 fused ops. Correctness includes non-power-of-2 shapes and boundary tiles, enforced by the harness.
**Forced learning:** shared-memory tiling, coalescing, occupancy, fusion, memory- vs compute-bound analysis.
**Learn:** *Prime/alongside:* PMPP ch. 1–6 (tiling, memory, performance) paced with the work; ch. 10 (reduction) for softmax/layernorm; GPU MODE lectures 4–9. *Consult:* Liger Kernel as "what good fused kernels look like."
**Hardware:** 4070 (Ada) — the transferable architecture.
**Trap:** correctness passing on one shape but failing on odd sizes/tails — and **this is wall #1** (see Operating cadence): when a kernel fights back for a week, drop to a smaller op; slow-but-correct counts as progress.
**Time:** ~5 weeks. Budget the full five; kernels fight back.

## B2 — Templated autotuning (deterministic "generation")
**Goal:** generate many kernel *variants* from a template and auto-select the fastest per shape.
**Ship:** `forge.autotune(op, shape)` — parameterize tile sizes / warps / stages / vectorization, sweep the space, cache the winner. Triton is the primary backend here. The cache is keyed by **(op, shape-bucket, GPU architecture)** — tunings don't transfer across architectures, and the key structure makes pretending otherwise impossible.
**Metric:** autotuned matches or beats your hand-written B1 kernels across a shape *distribution* (not one shape), with a persistent cache that makes the second run instant.
**Capstone:** Triton FlashAttention forward on the 4070 — ≥3× over the naive/math SDPA backend and ≥0.7× of PyTorch's flash SDPA backend across a shape sweep. Matching FA2 exactly isn't the point; being in its neighborhood with a kernel you wrote is.
**Forced learning:** the kernel design space, search over configs, Triton, per-shape specialization.
**Learn:** *Prime:* Triton tutorials (matmul/softmax/layernorm) — read, close, reimplement. *Deepen:* Dao 2022/2023 (FlashAttention 1–2).
**Trap:** autotuning to one shape and regressing on others. Tune over a distribution, cache per-shape.
**Time:** ~4 weeks.

## B3 — Agentic generation (LLM in the loop)
**Goal:** given a PyTorch reference, an agent writes a correct, faster kernel through a generate→compile→verify→profile→repair loop.
**Ship:** `forge.generate(reference_fn)` — a loop that prompts a frontier LLM (Claude API to start), compiles via B0, feeds back compiler errors + correctness diffs + profiler output, iterates up to N turns, and maintains persistent "lessons learned" memory per op-family, injected into prompts (you've built hybrid memory before — reuse that instinct). **Target Triton first**; CUDA C++ is the stretch backend.
**Metric:** on KernelBench Level 1 **plus a held-out 15-task set you author *before* building the loop** (so the loop can't be tuned to it): the loop's `fast_1` ≥ 2× the same model's one-shot `fast_1`. Track `fast_1`-per-version and **$-per-solved-kernel** — both go on the scoreboard.
**Budget:** API spend cap ~$100–150/month while active; log spend per run. Repair loops burn tokens fast.
**Forced learning:** agentic loop design, feedback grounding (compiler/profiler as reward signal), guarding against reward hacking in an *automated* setting.
**Learn:** *Consult:* CUDA-LLM (compilation+correctness+profiling feedback loop); Kevin (multi-turn RL for kernels). *Deepen:* re-read the Sakana post-mortem now that the adversary is real.
**Anti-hack rules:** the agent never sees harness source; every "pass" is re-verified on fresh random inputs and unseen shapes. Your B0 red-team gate is what stands between you and fooling yourself.
**Time:** ~5 weeks.

## B4 — Close the loop: your stack trains the kernel generator
**Goal:** finetune a small model with *your* `ember` stack to generate kernels, and have it drive `forge`.
**Framing (decided in the north-star section):** the base is a small open-weights **code** model (Qwen-Coder-class, 1.5–8B) — not your from-scratch 124M, which cannot write compiling kernels. What makes it "yours": your `ember/finetune.py`, your harvested dataset, your evals. A from-scratch-weights generator is the post-plan stretch.
**Ship:**
- A kernel dataset harvested from B3 traces: **only kernels that pass B0 on fresh inputs enter**; dedup near-identical solutions; rejection-sample to top-k by speedup per task.
- An SFT run (optional: DPO on fast-vs-slow kernel pairs) via `ember/finetune.py`, on the 4070 (LoRA/QLoRA).
- The finetuned model wired in as `forge.generate`'s default generator.
**Metric:** on the held-out task set: finetuned `fast_1` > the base model's `fast_1`, **and** an absolute floor of ≥30% correctness. Below the floor, the dataset is too small — go back to B3 and harvest more; don't ship the milestone on a technicality.
**Forced learning:** dataset construction from agent traces, rejection sampling, task-specialized finetuning, the full self-improvement data flywheel.
**Learn:** *Deepen:* KernelLLM (8B Llama finetuned for PyTorch→Triton — the proof this works at small scale, and the proof of why the base must be a code model).
**Trap:** training on your agent's mistakes. Only correct-and-fast kernels enter; every example is re-verified through B0 before it becomes a training row.
**Time:** ~4 weeks.

---

# The convergence & the improvement engine

Once B4 exists, the loop is live. "Keep improving" becomes a structured practice:

1. **`ember` gets `forge`'s kernels — on the 4070, where you train daily.** Drop fused kernels into the A4 ablation config's training path. Metric: tokens/sec on the standard A4 config, measured before and after each kernel lands. Every kernel win is a training-speed win you can graph. (No claims about H100 speedups you can't test — rented runs use stock fast paths.)
2. **`forge` gets `ember`'s brain.** Each improvement to your finetuning stack or dataset (A3/A4/B4) re-runs the B3/B4 scoreboard.
3. **The flywheel.** Better finetunes → better kernels → faster daily experiments → more A4 iterations per week → better models on the same budget. Log every turn of the crank.
4. **The stretch (post-plan):** scale ember past ~1B with code-heavy data, and swap your own pretrained weights in as the B4 base. That's the version of the loop where even the pretraining is yours.

**Two permanent scoreboards** (the anti-abandonment mechanism, quantified):
- `ember`: `experiments/LEADERBOARD.md` — loss at fixed token budget on the standard 4070 config.
- `forge`: `SCOREBOARD.md` — `fast_p` on KernelBench + your held-out set, plus $-per-solved-kernel.

If a week produced no movement on either number, that week stalled — and you'll know, because the number didn't move.

---

# Sequencing & timeline

`A0 → A1 → B0 → B1 → A2 (GPT-2) → B2 → A3 → B3 → B4 → close the loop → improve forever.`

Interleaving A and B is deliberate: B0/B1 slot in after A1 so you're not doing pure-model or pure-kernel work for months on end. (Note the A2 run does *not* wait for your kernels — it uses stock SDPA/`torch.compile`; your FlashAttention lands later, in B2, and proves itself on the 4070.)

At ~10 focused hrs/week:

| Milestone | Weeks | Cumulative |
|---|---|---|
| A0 — correctness spine | 3 | month 1 |
| A1 — tokenizer + data | 3 | month 1.5 |
| B0 — harness | 3 | month 2 |
| B1 — hand kernels *(wall #1)* | 5 | month 3.5 |
| A2 — GPT-2 124M *(wall #2)* | 4 | month 4.5 |
| B2 — autotuning + FA capstone | 4 | month 5.5 |
| A3 — finetuning stack | 4 | month 6.5 |
| B3 — agentic generation | 5 | month 7.5 |
| B4 — close the loop | 4 | month 8.5 |
| Convergence wiring | 2 | month 9 |

**~35–37 weeks ≈ 9 months nominal, 12 with life happening.** If your real cadence is 5 hrs/week, it's an 18-month plan — decide that consciously rather than discovering it in month 6.

---

# The learning system

The education hides inside the metrics — but only if you protect the *struggle*. The protocol per milestone:

1. **Prime** (one session, ≤2 hrs): watch/skim the primer listed in the milestone once, end-to-end, **no code-along, no notes-as-transcription**. You're loading the shape of the solution, not the solution.
2. **Build** to your spec. The reference stays closed.
3. **Stuck protocol** (this is the whole game): when blocked, struggle for a **45-minute timebox** first. Then write down, in one sentence, the specific question you can't answer. Open the reference, find the *idea* that answers it, **close the reference**, and implement from your written note. Never code side-by-side with a video or repo — that's transcription wearing a learning costume.
4. **Deepen** (after the metric is hit): read the listed papers properly. They land completely differently once you've built the thing — that's why they come last.
5. **Write it down:** each milestone ends with a build-log entry explaining what the metric took. The Nsight writeups in B1 and the experiment reports in A4 *are* the retention mechanism — no flashcards needed. Explaining the win is how you keep it.

**Reading map** (paced with milestones, not front-loaded):
- PMPP ch. 1–6 alongside B0/B1; ch. 10 (reduction) at softmax/layernorm; later chapters on demand.
- GPU MODE lectures 1–3 → B0; 4–9 → B1/B2.
- Papers are always Deepen-phase: Vaswani (A0), Sennrich (A1), GPT-2/Chinchilla (A2), InstructGPT/DPO/LoRA/QLoRA (A3), FlashAttention (B2), CUDA-LLM/Kevin/KernelLLM (B3/B4).

**Lab notebook:** a running `NOTES.md` per library — hypotheses, dead ends, numbers. Cheap to write, priceless in month 6.

---

# Operating cadence

**Weekly shape** (assumes ~10 hrs; adjust the count, keep the structure):
- 2 × 2-hr weeknight sessions — small, resumable tasks (tests, writeups, config sweeps).
- 1 × 4–6-hr weekend block — the deep work (kernels, training runs, debugging).
- **Monday 30-min review:** update the scoreboard, write one build-log paragraph (public — GitHub README or blog; "no progress" entries count and are the point), pick the *single* target for the week.

**Stall rule:** if a scoreboard number hasn't moved in 2 weeks, shrink the current task until something ships in one session. Momentum beats scope.

**The two walls — responses pre-committed now:**
- **B1 (kernels fight back):** a kernel that resists for a week gets demoted to a simpler op. Slow-but-correct is progress; profile before optimizing; the Nsight writeup of *why* it's slow counts as shipping.
- **A2 (the big run is finicky):** never debug on the 8× node. Every failure reproduces on the 2080S or a $3 single-GPU dry run first. The pre-flight checklist exists precisely for the moment you're tempted to skip it.

**Paused vs abandoned:** pausing is writing a dated note in the build log saying what's next and when you'll return. Anything else is abandoning. Make the state explicit.

---

# Budget

| Item | Est. |
|---|---|
| A2: 2–3 attempts on rented 8×H100 + dry runs | $150–300 |
| B3: API spend over ~2 active months (capped) | $200–300 |
| A3/B4 finetunes: local 4070 (LoRA/QLoRA), occasional cloud top-up | $0–100 |
| Misc (storage, dataset hosting, dry runs) | $50 |
| **Total** | **≈ $400–750** |

**Hardware roles:** 2080S (Turing, 8GB) = correctness dev + tiny overfits. 4070 (Ada, 12GB) = kernel dev, autotuning, finetunes, the A4 experiment loop — the machine the flywheel actually spins on. Rented 8×H100 = A2 only.

---

# Repo conventions

- `~/other/llm/` holds this plan + the build log; `ember/` and `forge/` are separate sibling repos, each with its own scoreboard file.
- Anything runnable gets a `bin/` entry point (`bin/ember train --config …`, `bin/forge bench …`) — no long commands from memory. Long-running services (TensorBoard, dashboards) get `start/stop/restart/status`.
- CI runs the cheap invariants on every push: A0's overfit test, A1's round-trip test, B0's anti-cheat suite. The tests that guard against fooling yourself are the ones that must never rot.

---

**Honest note:** this is a large, multi-quarter build on top of everything else you're carrying. The failure mode is not slowness — it's abandoning at B1 (kernels fight back) or A2 (the first real training run is finicky). Those are the two walls, and the pre-committed responses above are the ladder over them. Ship A2 (a GPT-2 you trained) and B1 (six hand-written kernels that beat eager, honestly benchmarked against `torch.compile`) and you already have two things almost nobody who "studies ML" ever produces. Everything after is the fun part: making them build each other.
