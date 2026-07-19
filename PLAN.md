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
- *Track 2 — a small open-weights model (1–2B, Llama-3.2-1B / Qwen-class):* where instruction lift is real and measurable. LoRA on a 1B fits on the 4070 (8GB laptop card); QLoRA beyond that. The full-FT row of the comparison table can't fit a 1–2B in 8GB (~16 bytes/param with AdamW) — do that row on a ~0.5B model with an 8-bit optimizer + activation checkpointing. This track is also deliberate practice for B4, which uses the same machinery.
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
**Framing (decided in the north-star section):** the base is a small open-weights **code** model (Qwen-Coder-class, 1.5–8B; on the 8GB 4070 that means QLoRA, comfortable at 1.5–3B — the 7–8B end is where the budgeted cloud top-up goes) — not your from-scratch 124M, which cannot write compiling kernels. What makes it "yours": your `ember/finetune.py`, your harvested dataset, your evals. A from-scratch-weights generator is the post-plan stretch.
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

# STAGE C — `ember/dist`: distributed training & the network it runs on

The loop closes on one GPU. This stage is about what happens when it can't — collectives, interconnects, parallelism schemes, fault tolerance — and it's the stage aimed straight at the systems interest. Everything ships inside `ember` as `ember/dist/`, built the same way as everything else: your implementation, parity-gated against the library version, with a number that can't be gamed. By the end, the Convergence stretch goal (pretraining your own >1B base for B4) stops being rhetorical, because a >1B pretrain *is* a distributed pretrain. EECS 491 gave you the vocabulary; here every abstraction has to earn its keep against a wire with measured bandwidth.

Two honesty clauses, decided now:

1. **Correctness is measured locally; performance is measured on rented matched hardware.** The 2080S + 4070 pair is heterogeneous (Turing vs Ada; both 8GB) — any job across them runs at the slow rank's pace, which makes it a fine correctness rig and a deliberately bad performance rig. Scoreboard perf numbers come only from rented matched GPUs (2×4090 spot ≈ $0.60–0.90/hr for C1–C3; 2-node instances for C4–C5). If the two cards live in separate boxes on your LAN, that's not a limitation — it's the C4 lab. A real network with real (miserable) bandwidth is the best teacher 1GbE will ever be.
2. **You will not beat NCCL or PyTorch DDP, and the metrics don't ask you to.** Every gate here is either a *parity gate* (match the library's numerics exactly; reach ≥90% of its throughput) or a *prediction gate* (write down what the performance model says *before* measuring; the metric is prediction error). Prediction error is this stage's un-gameable number, the way anti-cheat was B0's.

Standing rule: every milestone has a **2-process degenerate mode** (two ranks on one GPU, or gloo on CPU) so correctness never waits on hardware — and every distributed bug gets reproduced at the smallest world size that shows it before anything rented is touched. Pre-committed now, because the alternative is debugging NCCL hangs at $6/hr.

**Third permanent scoreboard:** `ember/dist/SCOREBOARD.md` — prediction error (%) per milestone, and MFU + tokens/sec/GPU at the C5 capstone config. Joins the Monday review with the other two.

## C0 — Collectives from scratch (the network is a device; learn it like one)
**Goal:** implement the core collectives yourself on point-to-point primitives, and build the comm benchmark the whole stage stands on.
**Ship:** `ember/dist/collectives.py` — ring all-reduce, all-gather, reduce-scatter, broadcast — built **twice**: first over raw TCP sockets (no torch, so nothing is hidden), then over `torch.distributed` P2P (gloo/NCCL). Plus `bin/ember bench-comm`: bandwidth/latency curves vs message size for every link you can touch (loopback, PCIe, LAN, rented NVLink), and a fitted **α–β cost model** (latency + bytes/bandwidth) per link.
**Metric:** (1) your all-reduce matches NCCL's output to written tolerance on randomized tensors across world sizes; (2) the ring hits ≥60% of measured link bandwidth at large message sizes; (3) the α–β fit predicts measured time within 15% across ≥3 orders of magnitude of message size. Report algbw vs busbw the way nccl-tests does — knowing the difference is part of the milestone.
**Forced learning:** ring vs tree algorithms, why all-reduce = reduce-scatter + all-gather, the latency regime vs the bandwidth regime, what NCCL actually is (transports, channels, protocols).
**Learn:** *Prime:* the Ultra-Scale Playbook's communication section + the classic Baidu ring-allreduce post. *Consult:* nccl-tests source for benchmark methodology. *Deepen:* NCCL docs on transports/protocols; "Demystifying NCCL" (2025).
**Trap:** benchmarking in the latency regime and calling it bandwidth. Small messages measure α, large ones measure β; the fit is only honest if the sweep covers both.
**Time:** ~3 weeks.

## C1 — DDP from scratch
**Goal:** your own data-parallel engine — semantically identical to single-GPU training, within 10% of PyTorch DDP's throughput.
**Ship:** `ember/dist/ddp.py` — autograd-hook-driven gradient sync with bucketing, comm/compute overlap, `no_sync`-style gradient accumulation, per-rank data sharding and seeding — wired into `ember/train.py` behind a flag.
**Metric:** (gate first, before any perf work) a 2-rank run's loss curve is statistically identical to a single-process run at the same effective batch — seeds fixed, tolerance written down in advance. Then: ≥90% of `DistributedDataParallel` throughput on rented matched 2×GPU, and an Nsight Systems trace showing all-reduce hidden under backward. The trace writeup is a deliverable, like B1's.
**Forced learning:** when autograd hooks fire, why DDP buckets gradients (~25MB) instead of syncing per-tensor, overlap scheduling, loss averaging vs summing, the uneven-last-batch problem.
**Learn:** *Consult:* the PyTorch DDP design note; picotron's `data_parallel.py` — diff against it only after yours runs, nanoGPT rules. *Deepen:* Li 2020 (the PyTorch DDP paper, VLDB — read after your version works; it's a catalog of everything you just hit).
**Trap:** "matches roughly" hiding a real bug — seed/shard/dropout mistakes look like noise for a thousand steps. The equivalence gate exists so *close* is never mistaken for *correct*.
**Time:** ~3 weeks.

## C2 — ZeRO from scratch (memory is the real currency)
**Goal:** shard optimizer state and gradients (ZeRO-1/2) with a written memory model the hardware confirms; ZeRO-3 (param sharding) as stretch.
**Ship:** `ember/dist/zero.py`, plus a memory-accounting note — predicted bytes/param for params, grads, optimizer state, and mixed-precision master copies at each ZeRO stage — **written before implementing**.
**Metric:** (1) predicted peak VRAM within 10% of `torch.cuda.max_memory_allocated` across configs; (2) ZeRO-2 trains a config that demonstrably OOMs under your C1 DDP on the same GPU; (3) loss parity with DDP (same gate as C1).
**Forced learning:** where training memory actually goes, reduce-scatter-based gradient sharding, per-stage communication volume vs plain DDP, why ZeRO-3 pays an extra all-gather every step.
**Learn:** *Prime:* Ultra-Scale Playbook ZeRO chapter. *Deepen:* Rajbhandari 2020 (ZeRO); Zhao 2023 (PyTorch FSDP).
**Trap:** measuring memory with `nvidia-smi` — the caching allocator makes it lie. Use torch's allocator stats, and account for master weights explicitly or the model double-counts.
**Time:** ~3 weeks.

## C3 — Tensor & pipeline parallelism
**Goal:** Megatron-style TP for the MLP and attention blocks, and pipeline parallelism with a 1F1B schedule — both proven equivalent to the single-GPU model.
**Ship:** `ember/dist/tp.py` (column/row-parallel linears; the two-all-reduce transformer block), `ember/dist/pipeline.py` (microbatching; GPipe schedule first, then 1F1B).
**Metric:** (1) TP=2 matches single-GPU outputs to numerical tolerance — TP is a refactor of the same math and must *match*, not approximate; (2) measured pipeline bubble fraction within a few points of the (p−1)/(m+p−1) prediction across ≥3 microbatch counts; (3) a comm-volume table — bytes per layer per step for DP vs TP vs PP — validated against traced traffic.
**Forced learning:** which matmuls split column-wise vs row-wise and why the block needs exactly two all-reduces, microbatching and activation memory vs bubble tradeoffs, why TP dies on slow links while PP tolerates them.
**Learn:** *Prime:* Ultra-Scale Playbook TP + PP chapters. *Consult:* picotron's `tensor_parallel.py`/`pipeline_parallel.py`, same diff-after rule. *Deepen:* Shoeybi 2019 (Megatron-LM); Huang 2019 (GPipe); Narayanan 2021 (the 1F1B analysis).
**Hardware:** rented matched 2–4×GPU for perf; the 2-process degenerate mode for all correctness work.
**Trap:** testing only dimensions that divide evenly by the parallel degree — B1's power-of-2 disease wearing a new coat. Odd head counts and vocab remainders go in the gate.
**Time:** ~4 weeks.

## C4 — Multi-node: the actual network
**Goal:** run your stack across physically separate machines, and predict its throughput from the C0 model *before* measuring it.
**Ship:** `bin/ember launch` (rendezvous, env plumbing, the NCCL debug knobs you'll need); an MFU calculator; and the **bandwidth-ladder writeup** — measured α/β for every rung you can touch: intra-box PCIe, your LAN, rented NVLink, rented inter-node fabric (IB/RoCE vs plain TCP).
**Metric:** a pre-registered prediction: tokens/sec for a 2-node DDP+ZeRO run, written down before launch, measured within 20%. Then find one real bottleneck in the trace (bucket size vs link latency, stragglers, host networking) and fix it — before/after numbers or it didn't happen.
**Forced learning:** TCP vs RDMA data paths, GPUDirect, what `NCCL_DEBUG`/`NCCL_SOCKET_IFNAME` actually reveal, rendezvous and launchers, why gradient compression research exists.
**Learn:** *Consult:* Stas Bekman's ML Engineering book (networking chapters) — field notes for exactly this. *Deepen:* an RDMA/verbs primer; re-read "Demystifying NCCL" now that you've been burned by it.
**Trap:** if the local lab is two boxes on gigabit Ethernet, the numbers will be terrible — that's the curriculum, not a failure. The win is the model predicting *exactly how* terrible. Rent IB-connected nodes to see the other end of the ladder; don't try to tune your way past physics.
**Time:** ~3 weeks.

## C5 — Fault tolerance + the capstone run
**Goal:** training that survives death, then the stage capstone: a pre-registered multi-node run hitting an MFU target with your stack end to end.
**Ship:** sharded checkpointing with kill-tested resume; restart-without-babysitting (a rank dies → the job resumes from the last checkpoint on its own); the capstone run report.
**Metric:** (1) `kill -9` a rank at a randomly chosen step: the run resumes automatically and final loss matches an uninterrupted control within written tolerance; (2) **capstone:** a rented 2-node run training a ~350M–1B `ember` config with your DDP+ZeRO hybrid, hitting an MFU target you pick and justify beforehand from your roofline + α–β model (~35–40% is the honest neighborhood for that setup). Pre-registration *is* the metric — hitting a number you predicted is the whole stage in one run.
**Forced learning:** MFU math from first principles, checkpoint sharding, failure modes at scale, why elastic training is genuinely hard.
**Learn:** *Consult:* torchrun-elastic and torchft design docs. *Deepen:* the capstone report you write — this milestone's Deepen is your own postmortem.
**Trap:** "it resumed" without proving equivalence. A resume that silently skips or replays a data shard passes the eye test and corrupts the run; the control-run comparison is the gate.
**Time:** ~3 weeks.

**What C unlocks:** with C0–C5 shipped, the Convergence stretch — scaling `ember` past 1B and swapping your own pretrained weights into B4 — is an engineering exercise instead of a wish. That's the version of the loop where even the pretraining is yours, running on a parallelism stack that is also yours.

---

# Sequencing & timeline

`A0 → A1 → B0 → B1 → A2 (GPT-2) → B2 → A3 → B3 → B4 → close the loop → C0 → C1 → C2 → C3 → C4 → C5 → improve forever.`

Stage C is last on purpose: it needs a working `ember` to parallelize, and by then A2 has already made you a *user* of torchrun/DDP — C is where you take the lid off. If the systems itch gets unbearable earlier, C0 (collectives + the comm benchmark) is self-contained and can slot in any time after A2 without disturbing the rest.

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
| C0 — collectives + comm bench | 3 | month 9.75 |
| C1 — DDP from scratch | 3 | month 10.5 |
| C2 — ZeRO | 3 | month 11.25 |
| C3 — TP + PP | 4 | month 12.25 |
| C4 — multi-node *(wall #3)* | 3 | month 13 |
| C5 — fault tolerance + capstone | 3 | month 13.75 |

**~54–56 weeks ≈ 13–14 months nominal, 17–18 with life happening.** If your real cadence is 5 hrs/week, it's a two-year plan — decide that consciously rather than discovering it in month 6.

---

# The learning system

The education hides inside the metrics — but only if you protect the *struggle*. The protocol per milestone:

1. **Prime** (one session, ≤2 hrs): watch/skim the primer listed in the milestone once, end-to-end, **no code-along, no notes-as-transcription**. You're loading the shape of the solution, not the solution.
2. **Build** to your spec. The reference stays closed.
3. **Stuck protocol** (this is the whole game): when blocked, struggle for a **45-minute timebox** first. Then write down, in one sentence, the specific question you can't answer. Open the reference, find the *idea* that answers it, **close the reference**, and implement from your written note. Never code side-by-side with a video or repo — that's transcription wearing a learning costume.
4. **Deepen** (after the metric is hit): read the listed papers properly. They land completely differently once you've built the thing — that's why they come last.
5. **Write it down:** each milestone ends with a build-log entry explaining what the metric took. The Nsight writeups in B1 and the experiment reports in A4 *are* the retention mechanism — no flashcards needed. Explaining the win is how you keep it.

**Reading map** (paced with milestones, not front-loaded — every item below is linked, with its Prime/Consult/Deepen slot, in [LEARNING.md](LEARNING.md)):
- PMPP ch. 1–6 alongside B0/B1; ch. 10 (reduction) at softmax/layernorm; later chapters on demand.
- GPU MODE lectures 1–3 → B0; 4–9 → B1/B2; the NCCL/distributed lectures → C0/C1.
- The Ultra-Scale Playbook (HuggingFace) is Stage C's spine — one chapter primed per milestone (comms → C0, DP → C1, ZeRO → C2, TP/PP → C3), never read ahead of the build. "How to Scale Your Model" (the DeepMind scaling book) for the roofline/comm mental math; Bekman's ML Engineering networking chapters at C4.
- Papers are always Deepen-phase: Vaswani (A0), Sennrich (A1), GPT-2/Chinchilla (A2), InstructGPT/DPO/LoRA/QLoRA (A3), FlashAttention (B2), CUDA-LLM/Kevin/KernelLLM (B3/B4), PyTorch DDP (C1), ZeRO/FSDP (C2), Megatron/GPipe/1F1B (C3), Demystifying NCCL (C0/C4).

**Lab notebook:** a running `NOTES.md` per library — hypotheses, dead ends, numbers. Cheap to write, priceless in month 6.

---

# Operating cadence

**Weekly shape** (assumes ~10 hrs; adjust the count, keep the structure):
- 2 × 2-hr weeknight sessions — small, resumable tasks (tests, writeups, config sweeps).
- 1 × 4–6-hr weekend block — the deep work (kernels, training runs, debugging).
- **Monday 30-min review:** update the scoreboard, write one build-log paragraph (public — GitHub README or blog; "no progress" entries count and are the point), pick the *single* target for the week.

**Stall rule:** if a scoreboard number hasn't moved in 2 weeks, shrink the current task until something ships in one session. Momentum beats scope.

**The three walls — responses pre-committed now:**
- **B1 (kernels fight back):** a kernel that resists for a week gets demoted to a simpler op. Slow-but-correct is progress; profile before optimizing; the Nsight writeup of *why* it's slow counts as shipping.
- **A2 (the big run is finicky):** never debug on the 8× node. Every failure reproduces on the 2080S or a $3 single-GPU dry run first. The pre-flight checklist exists precisely for the moment you're tempted to skip it.
- **C4 (multi-node hangs give you no stack trace, just a bill):** every distributed bug reproduces at the smallest world size that shows it — 2 processes on one box, gloo on CPU if it still reproduces there — before anything rented is touched. `NCCL_DEBUG=INFO` from the first run, not after the first hang.

**Paused vs abandoned:** pausing is writing a dated note in the build log saying what's next and when you'll return. Anything else is abandoning. Make the state explicit.

---

# Budget

| Item | Est. |
|---|---|
| A2: 2–3 attempts on rented 8×H100 + dry runs | $150–300 |
| B3: API spend over ~2 active months (capped) | $200–300 |
| A3/B4 finetunes: local 4070 (LoRA/QLoRA), occasional cloud top-up | $0–100 |
| C1–C3: rented matched 2×4090 spot, dev + perf hours | $50–100 |
| C4–C5: rented 2-node time (incl. IB nodes + the capstone run) | $100–200 |
| Misc (storage, dataset hosting, dry runs) | $50 |
| **Total** | **≈ $550–1050** |

**Hardware roles:** 2080S (Turing, 8GB) = correctness dev + tiny overfits. 4070 laptop (Ada, 8GB) = kernel dev, autotuning, finetunes, the A4 experiment loop — the machine the flywheel actually spins on. Laptop caveat: GPU clocks drift with thermals and usually can't be locked — pin the power profile and log clock speed during any benchmark run, or the timing numbers are noise. Rented 8×H100 = A2 only. Stage C: the local pair = distributed *correctness* rig (heterogeneous — never quote perf from it) and, if the cards live in separate boxes, the C4 home-LAN networking lab; rented matched 2×4090 = C1–C3 perf rig; rented 2-node = C4/C5.

---

# Repo conventions

- `~/other/llm/` holds this plan + the build log; `ember/` and `forge/` are separate sibling repos, each with its own scoreboard file.
- Anything runnable gets a `bin/` entry point (`bin/ember train --config …`, `bin/forge bench …`) — no long commands from memory. Long-running services (TensorBoard, dashboards) get `start/stop/restart/status`.
- CI runs the cheap invariants on every push: A0's overfit test, A1's round-trip test, B0's anti-cheat suite. The tests that guard against fooling yourself are the ones that must never rot.

---

**Honest note:** this is a large, multi-quarter build on top of everything else you're carrying. The failure mode is not slowness — it's abandoning at B1 (kernels fight back) or A2 (the first real training run is finicky). Those are the two walls, and the pre-committed responses above are the ladder over them. Ship A2 (a GPT-2 you trained) and B1 (six hand-written kernels that beat eager, honestly benchmarked against `torch.compile`) and you already have two things almost nobody who "studies ML" ever produces. Everything after is the fun part: making them build each other — and Stage C is the part pointed at where your interest actually lives, the systems underneath the training run. When the middle of the plan drags, that's the stage you're walking toward.
