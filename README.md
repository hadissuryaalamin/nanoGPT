# gpt2-rocstories-finetuning

Training a small GPT to write coherent five-sentence stories — from scratch, under a
32M-parameter ceiling, through a three-stage fine-tuning curriculum.

> **Built on [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)** (MIT licensed —
> `LICENSE` is his, unchanged). The training loop (`train.py`) and the model definition
> (`model.py`) are his work, and I have kept them close to upstream on purpose.
>
> **What is mine:** the data pipelines under `data/rocstories*` and `data/tinystories`,
> the staged training configs in `config/`, the discourse-tag SFT format, the evaluation
> and model-comparison tooling (`eval.py`, `eval_rocstories.py`, `compare_models.py`),
> and the experiments and results reported below.

---

## The problem

ROCStories is a corpus of five-sentence everyday stories. The task: get a language model
small enough to train on a single GPU to produce stories that are *fluent*, *exactly five
sentences*, and *structured like a story* — setup, conflict, reaction, attempt,
resolution — rather than five sentences that merely follow each other.

The constraint was a **32M-parameter ceiling**, so most of the work is about what you can
buy with a training curriculum rather than with scale.

## The model

| | |
|---|---|
| Layers | 6 |
| Heads | 6 |
| Embedding dim | 384 |
| Context (`block_size`) | 256 |
| Vocab | 50257 (GPT-2 BPE, `tiktoken`) |
| Biases | none (`bias = False`) |
| Parameters | **29.92M** as nanoGPT reports it (30.02M including position embeddings; token embedding and output head are tied) |
| Precision | bfloat16 |

## The curriculum

Each stage starts from the previous stage's checkpoint. The ordering is deliberate:
learn the domain, then borrow fluency from cleaner data, then learn structure explicitly,
then remove the scaffolding that taught it.

| Stage | From | Data | Goal |
|---|---|---|---|
| **Pretrain** | scratch | ROCStories | Learn the domain and its register. Produces `out-rocstories/25_62.pt`. |
| **1** | pretrain | TinyStories | Buy fluency and coherence from higher-quality synthetic story data. `config/finetune_tinystories.py` |
| **2** | stage 1 | ROCStories, discourse-tagged | Teach explicit narrative structure with tags:<br>`<\|s1\|>` setup `<\|s2\|>` conflict `<\|s3\|>` reaction `<\|s4\|>` attempt `<\|s5\|>` resolution. `config/finetune_rocstories_discourse.py` |
| **3** | stage 2 | ROCStories, clean | Unlearn the tag *format* while keeping the structure it taught, so inference needs no special tokens. Very low LR (`1e-5`, no decay) so fluency is not disturbed. `config/finetune_rocstories_clean.py` |

Stage 3 is the point of the whole design. The discourse tags are a teaching aid, not a
deliverable: the final model should write structured stories in plain text, promptable
with nothing but a first sentence.

## Results

**Test perplexity on ROCStories: 25.62 → 25.18** (pretrain → final; see
`out-rocstories-clean/best_ppl.txt`).

That is a deliberately modest number and worth reading honestly: a 0.44 perplexity
improvement is small, and perplexity is not what this curriculum was chiefly buying.
The change that matters is qualitative and visible in `compare_outputs/summary.txt`,
which samples all four checkpoints on the same prompts.

Given the prompt *"Tom decided to cook dinner for his friends."*:

- **Pretrain** — on-domain but semantically adrift:
  > They all ordered a large salad. They got the ingredients together. They all ate it and never had so much food. Tom decided to make a salad for himself.

- **Stage 1 (TinyStories)** — fluent, but it has taken on the children's-story register and will not stop at five sentences:
  > He put on a big white paper, some vegetables and some meat. He made a peanut paper for his friends, his dog, and his dog. […] They were happy and they all celebrated and had a fun time.

- **Stage 2 (discourse)** — correct five-part structure, but leaks its scaffolding:
  > `<\|s2\|>` He was just about to finish cooking when he heard a loud pop. `<\|s3\|>` He panicked and ran for cover. `<\|s4\|>` The smoke alarm went off. `<\|s5\|>` He called the police and everyone was sentenced to prison.

- **Final** — five sentences, no tags, holds its subject:
  > He decided to make it himself. He got all the ingredients together. When he got the meat he found some sauce. Tom loved it.

So the curriculum did what it was designed to do: Stage 1 bought fluency at the cost of
format, Stage 2 imposed format, and Stage 3 kept the format while dropping the tokens
that taught it.

## Honest limitations

- **It is a 30M-parameter model.** Outputs stay locally coherent and lose global sense.
  Stage 2 samples drift between characters (a story opening on *Emily* continues about
  *Rita*); final-stage samples produce lines like *"made her jog without a bleeding"*.
- **Perplexity is a weak instrument here** and moved very little. The evidence for the
  curriculum is the side-by-side sampling, which is qualitative and unblinded — I chose
  the prompts.
- **No held-out human evaluation**, no automatic structure metric (e.g. checking that
  exactly five sentences are emitted), and no ablation isolating which stage contributed
  what. Stage 1's fluency gain and Stage 2's structure gain are confounded.
- Stage 3 is only ~1500 iterations at a low learning rate; how completely the tag format
  is unlearned is not measured, only observed.

## Reproducing

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Prepare data (each dataset has its own `prepare.py`):

```sh
python data/rocstories/prepare.py
python data/tinystories/prepare.py
python data/rocstories_discourse/prepare.py
```

Then run the stages in order. Each config's header documents the checkpoint copy it
expects first — the stages chain by seeding the next `out_dir` with the previous
checkpoint:

```sh
python train.py config/train_rocstories.py                  # pretrain
python train.py config/finetune_tinystories.py              # stage 1
python train.py config/finetune_rocstories_discourse.py     # stage 2
python train.py config/finetune_rocstories_clean.py         # stage 3
```

Evaluate and sample:

```sh
python eval.py --init_from=resume --out_dir=out-rocstories-clean \
               --input_file=data/rocstories/test.txt
python sample.py --out_dir=out-rocstories-clean --start="Tom decided to cook dinner."
python compare_models.py        # regenerates compare_outputs/summary.txt
```

## What is in here

| Path | |
|---|---|
| `config/` | the staged curriculum — each file's header explains its stage |
| `data/rocstories*`, `data/tinystories/` | dataset preparation and tokenisation |
| `eval.py`, `eval_rocstories.py` | perplexity evaluation |
| `compare_models.py`, `compare_outputs/` | side-by-side sampling across all four checkpoints |
| `log/` | per-stage training logs (`PT-ROCStories-*`, `FT-TinyStories`, `FT-Discourse`, `FT-Final`) |
| `out-rocstories-clean/best_ppl.txt` | the final reported perplexity |
| `train.py`, `model.py`, `sample.py`, `bench.py` | upstream nanoGPT, essentially unmodified |

## Credit

The scaffolding is Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT),
used under the MIT licence retained in `LICENSE`. Datasets: ROCStories (Mostafazadeh et
al.) and [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (Eldan &
Li).
