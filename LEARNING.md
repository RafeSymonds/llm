# LEARNING.md — the material shelf

Every resource PLAN.md names, with links, organized by milestone and by *slot* (Prime / Consult / Deepen — see PLAN.md → The learning system). The rules travel with the shelf:

- **Prime** = one pass, ≤2 hrs, before building. No code-along, no transcription. You're loading the shape of the solution.
- **Consult** = stuck protocol only: 45-min timebox → write the one-sentence question → open the reference, find the *idea* → **close it** → implement from your note.
- **Deepen** = read properly only after the milestone's metric is hit. Papers land differently once you've built the thing.

Everything below is free except PMPP (buy the book).

---

## Stages A & B — quick links

| Milestone | Slot | Material |
|---|---|---|
| A0 | Prime | Karpathy, ["Let's build GPT: from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| A0 | Consult | [nanoGPT](https://github.com/karpathy/nanoGPT) (diff only after yours runs) |
| A0 | Deepen | Vaswani 2017, [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) |
| A1 | Prime | Karpathy, ["Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE) |
| A1 | Deepen | Sennrich 2016, [BPE](https://arxiv.org/abs/1508.07909) |
| A2 | Prime | Karpathy, ["Let's reproduce GPT-2 (124M)"](https://www.youtube.com/watch?v=l8pRSuU81PU) |
| A2 | Consult | [llm.c](https://github.com/karpathy/llm.c) + nanoGPT configs |
| A2 | Deepen | [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf); Hoffmann 2022, [Chinchilla](https://arxiv.org/abs/2203.15556) |
| A3 | Consult | [nanochat](https://github.com/karpathy/nanochat) (full-lifecycle reference) |
| A3 | Deepen | [InstructGPT](https://arxiv.org/abs/2203.02155); [DPO](https://arxiv.org/abs/2305.18290); [LoRA](https://arxiv.org/abs/2106.09685); [QLoRA](https://arxiv.org/abs/2305.14314) |
| A4 | Consult | Llama / Mistral / Qwen papers, per experiment, on demand |
| B0 | Prime | [GPU MODE lectures](https://github.com/gpu-mode/lectures) 1–3 ([YouTube](https://www.youtube.com/@GPUMODE)); [KernelBench](https://github.com/ScalingIntelligence/KernelBench) paper, harness section |
| B0 | Deepen | [Sakana AI CUDA Engineer post-mortem](https://sakana.ai/ai-cuda-engineer/) — read as a defender |
| B1 | Alongside | **PMPP** (Hwu/Kirk/El Hajj, 4th ed.) ch. 1–6, ch. 10; GPU MODE lectures 4–9 |
| B1 | Consult | [Liger Kernel](https://github.com/linkedin/Liger-Kernel) |
| B2 | Prime | [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) (read → close → reimplement) |
| B2 | Deepen | [FlashAttention](https://arxiv.org/abs/2205.14135); [FlashAttention-2](https://arxiv.org/abs/2307.08691) |
| B3 | Consult | CUDA-LLM and Kevin papers (search titles on arXiv: "CUDA-LLM", "Kevin: Multi-Turn RL") |
| B3/B4 | Deepen | Sakana post-mortem (re-read); [KernelLLM](https://huggingface.co/facebook/KernelLLM) |

---

## Stage C — the full shelf

### The four stage-wide resources

1. **[The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)** (HuggingFace, 2025) — the spine of the stage. Every parallelism scheme, ZeRO, comm/compute overlap, distilled from ~4,000 scaling experiments. Also available [as a PDF](https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/main/The_Ultra-Scale_Playbook_Training_LLMs_on_GPU_Clusters.pdf) if the interactive page tempts you to click around instead of reading. **One chapter per milestone, primed right before that milestone — never read ahead of the build.**
2. **[How to Scale Your Model](https://jax-ml.github.io/scaling-book/)** (DeepMind) — the mental-math book: rooflines, comm cost accounting, "how long should this take" arithmetic. JAX-flavored but the math is framework-free. Read the roofline chapter at C0; return for MFU math at C4/C5.
3. **[picotron](https://github.com/huggingface/picotron)** — the nanoGPT of this stage: DP/TP/PP/CP each in a single file under ~300 lines. This is your *diff-after-yours-runs* reference for C1–C3, same rule as nanoGPT in A0. There is also [picotron_tutorial](https://github.com/huggingface/picotron_tutorial), a step-by-step video playlist + companion repo — **maximum transcription hazard**: it exists to be coded along with, which is exactly what you don't do. Stuck-protocol access only.
4. **[Stas Bekman, *Machine Learning Engineering*](https://github.com/stas00/ml-engineering)** — field notes from real training runs. The [network chapter](https://github.com/stas00/ml-engineering/tree/master/network) is C4's primer; the debug chapter earns its keep the first time a run hangs.

### C0 — Collectives from scratch
- **Prime:** Ultra-Scale Playbook, appendix **A0 "Parallel Programming Crash Course"** (broadcast → gather → all-reduce, with diagrams); then Gibiansky, ["Bringing HPC Techniques to Deep Learning"](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) — *the* ring-allreduce post (2017, Baidu), still the cleanest derivation of why ring = 2(N−1)/N × optimal.
- **Consult:** [Writing Distributed Applications with PyTorch](https://pytorch.org/tutorials/intermediate/dist_tuto.html) — the send/recv API you'll build on. Warning: it implements a ring all-reduce near the end; don't read that section until yours works. [nccl-tests](https://github.com/NVIDIA/nccl-tests) — its [PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) is the canonical algbw/busbw definition your metric cites. [GPU MODE Lecture 17: GPU Collective Communication (NCCL)](https://github.com/gpu-mode/lectures).
- **Deepen:** ["Demystifying NCCL"](https://arxiv.org/abs/2507.04786) (Hu et al. 2025, with NCCL's own Sylvain Jeaugey as coauthor) — protocols (Simple/LL/LL128), channels, ring vs tree; the [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) alongside.

### C1 — DDP from scratch
- **Prime:** Ultra-Scale Playbook, **Data Parallelism** chapter (first half — stop before ZeRO; that's C2's primer).
- **Consult:** the [PyTorch DDP design note](https://pytorch.org/docs/stable/notes/ddp.html) — bucketing and hook mechanics, in prose; picotron `data_parallel.py` after yours runs.
- **Deepen:** Li et al. 2020, ["PyTorch Distributed: Experiences on Accelerating Data Parallel Training"](https://arxiv.org/abs/2006.15704) (VLDB) — a catalog of every problem you just hit, with their measurements.

### C2 — ZeRO from scratch
- **Prime:** Ultra-Scale Playbook, **Data Parallelism** chapter, ZeRO sections (memory-anatomy diagrams — the ones your memory model must reproduce from first principles).
- **Consult:** picotron; the [PyTorch memory-stats docs](https://pytorch.org/docs/stable/torch_cuda_memory.html) for honest VRAM measurement (`torch.cuda.memory` + the memory snapshot visualizer).
- **Deepen:** Rajbhandari et al. 2020, [ZeRO](https://arxiv.org/abs/1910.02054); Zhao et al. 2023, [PyTorch FSDP](https://arxiv.org/abs/2304.11277).

### C3 — Tensor & pipeline parallelism
- **Prime:** Ultra-Scale Playbook, **Tensor Parallelism** chapter, then **Pipeline Parallelism** chapter (the bubble-fraction algebra your metric checks is derived there).
- **Consult:** picotron `tensor_parallel.py` / `pipeline_parallel.py`; [GPU MODE Lecture 13: Ring Attention](https://github.com/gpu-mode/lectures) if you take the context-parallel stretch.
- **Deepen:** Shoeybi et al. 2019, [Megatron-LM](https://arxiv.org/abs/1909.08053); Huang et al. 2019, [GPipe](https://arxiv.org/abs/1811.06965); Narayanan et al. 2021, [Efficient Large-Scale Training](https://arxiv.org/abs/2104.04473) (the 1F1B analysis).

### C4 — Multi-node: the actual network
- **Prime:** Bekman, [network chapter](https://github.com/stas00/ml-engineering/tree/master/network) — intra- vs inter-node speeds, what NICs/fabrics actually deliver, how to benchmark a cluster you just rented.
- **Consult:** Bekman's [`torch-distributed-gpu-test.py`](https://github.com/stas00/ml-engineering) diagnostic; [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) (`NCCL_DEBUG`, `NCCL_SOCKET_IFNAME`, and friends); [torchrun docs](https://pytorch.org/docs/stable/elastic/run.html) for rendezvous.
- **Deepen:** re-read ["Demystifying NCCL"](https://arxiv.org/abs/2507.04786) — the transport/protocol sections read completely differently after a week of real multi-node; RDMA/verbs material linked from Bekman's network chapter if the fabric layer hooks you.

### C5 — Fault tolerance + capstone
- **Consult:** [torch.distributed.checkpoint (DCP)](https://pytorch.org/docs/stable/distributed.checkpoint.html) docs — sharded checkpoint format and resharding; [torchft](https://github.com/pytorch/torchft) design docs for what production fault tolerance looks like; [GPU MODE Lecture 70: Fault-tolerant communication collectives](https://github.com/gpu-mode/lectures); [TorchTitan](https://github.com/pytorch/torchtitan) (and GPU MODE Lecture 39) as the reference for what a full production-grade version of your whole C stack looks like.
- **Deepen:** the MFU definition at the source — [PaLM](https://arxiv.org/abs/2204.02311), Appendix B; then your own capstone report, which is this milestone's real Deepen.

---

## Acquisition checklist

- [ ] Buy PMPP 4th ed. (the one non-free item; needed by B0)
- [ ] Bookmark the Ultra-Scale Playbook + download the PDF
- [ ] Clone picotron + star picotron_tutorial (but see the hazard warning above)
- [ ] Clone [gpu-mode/lectures](https://github.com/gpu-mode/lectures) — slides/code live there, videos on [YouTube](https://www.youtube.com/@GPUMODE)
- [ ] Clone [stas00/ml-engineering](https://github.com/stas00/ml-engineering)
- [ ] Papers: pull PDFs as each Deepen phase arrives, not before — front-loading them is how reading substitutes for building
