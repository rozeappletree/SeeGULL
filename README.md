# **`SeeGULL:`** Can LLMs Read Human Gullibility?

<img width="360" alt="seagull" src="https://github.com/user-attachments/assets/1e51b173-a036-498c-8744-52fb3eaa87da" />

An LLM changes its answers when you give it a persona. This project asks whether it also
changes them based on an **unstated** property of the person it is talking to — specifically,
**how gullible that person is** — and whether that property is legible in the model's own
activations.

The answer so far is **yes**: a linear probe on Llama-2-13B-Chat activations separates
high- from low-gullibility human speakers at **88–99% accuracy**, and the signal survives a
full distribution shift.

> **Read first:** [Why is this question important?](docs/mats-applications-answers/2-why-interesting.md)
> · [Full project report (LaTeX)](docs/mats-applications-answers/project_report.tex)

---

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/probe-accuracy-dark.png">
  <img alt="Reading-probe accuracy across four evaluation sets: 99.76% on same-distribution holdout, 97.88% on Defense-484, 88.39% on Hard-333, 88.38% on Ardulous-66" src="assets/probe-accuracy-light.png">
</picture>

| Eval set | n | Accuracy | Macro F1 | Relationship to training data |
|---|---:|---:|---:|---|
| Holdout | 842 | **99.76%** | 0.998 | Same distribution |
| Defense-484 | 2,880 | **97.88%** | 0.979 | Same pipeline, non-hard-negative claims |
| Hard-333 | 2,119 | **88.39%** | 0.884 | Different distribution, different generator model |
| Ardulous-66 | 413 | **88.38%** | 0.884 | Hardest subset, most distinct distribution |

The number that matters is the last one. Accuracy **degrades gracefully** (99.8% → 88.4%) as the
eval set moves away from the training distribution, rather than collapsing toward chance —
which is what you would expect if the probe had only memorised the stylistic fingerprint of one
generation pipeline. Spot-checking the Ardulous-66 errors, most sit on genuinely ambiguous
conversations (a user defending a plausible-but-wrong answer with reasonable arguments), not on
obvious failures.

Probe: `NousResearch/Llama-2-13b-chat-hf`, layer 38, trained on v0.3 `train/` only.

---

## New: the v1 dataset

`datasets_deepseek_gullibility_v1/` — **29,621 conversations**, the largest release here and
**7.4×** the corpus the current best probe was trained on. Generated with
`deepseek/deepseek-v4-pro-0813` across **16,414 distinct topics**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/dataset-growth-dark.png">
  <img alt="Dataset scale by release: v0.1 has 170 conversations, v0.3 has 3,992, v1 has 29,621" src="assets/dataset-growth-light.png">
</picture>

**It is not a drop-in replacement for v0.3 yet.** Two things to know before you train on it:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/v1-balance-dark.png">
  <img alt="v1 class composition: 16,955 low-gullibility (57.2%) versus 12,666 high-gullibility (42.8%)" src="assets/v1-balance-light.png">
</picture>

1. **The classes are unbalanced** — 16,955 low vs 12,666 high (57.2 / 42.8). v0.3 was cleaned to
   exact balance; v1 is raw generator output. Subsample, weight the loss, or run it through
   [`nb/dedupe.ipynb`](nb/dedupe.ipynb) first.
2. **It has not been deduplicated or leak-checked.** v0.3's cleaning pass dropped near-duplicates,
   stub turns and attribute leaks (287 conversations in total). The same pass has not been run
   on v1.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/v1-lengths-dark.png">
  <img alt="v1 conversation length distributions by class overlap heavily, but low-gullibility conversations have a median of 118 words versus 102 for high" src="assets/v1-lengths-light.png">
</picture>

There is also a **length confound worth controlling for**: the distributions overlap heavily, but
low-gullibility conversations run longer (median 118 words vs 102). A probe that keys on length
alone would get some of the way — which is exactly why a TF-IDF / length baseline is on the
open-questions list below.

<details>
<summary><strong>v1 at a glance</strong></summary>

| | |
|---|---|
| Conversations | 29,621 (16,955 low / 12,666 high) |
| Generator | `deepseek/deepseek-v4-pro-0813` |
| Distinct topics | 16,414 |
| Turns | 6 in 89% of conversations (range 5–10) |
| Length | median 111 words, mean 116, range 34–516 |
| Generated | 2026-09-11 → 2026-09-12 |
| Cleaned? | **No** — raw generator output |

Each conversation ships as a pair of files:

```
conversation_<i>_gullibility_<high|low>.txt     full HUMAN:/ASSISTANT: transcript
conversation_<i>_gullibility_<high|low>.json    level, topic, model, seed, num_turns, call_index
```

</details>

---

## All dataset releases

| Dataset | n | Balanced | Cleaned | Role |
|---|---:|:---:|:---:|---|
| [`datasets_deepseek_gullibility_v1`](datasets_deepseek_gullibility_v1/) | 29,621 | ✗ | ✗ | **New.** Next training run |
| [`datasets_deepseek_gullibility_v0_3.deduped.3992`](datasets_deepseek_gullibility_v0_3.deduped.3992/) | 3,992 | ✓ | ✓ | Trains the current best probe |
| `datasets_defense_484` (in [`mats12/`](mats12/)) | 2,880 | ✓ | — | Eval — non-hard-negative claims |
| `datasets_hard_333` (in [`mats12/`](mats12/)) | 2,119 | ✓ | — | Eval — hard negatives |
| `datasets_ardulous_66` (in [`mats12/`](mats12/)) | 413 | ✓ | — | Eval — hardest subset |
| `datasets_regular_gullibility_170` (in [`mats12/`](mats12/)) | 170 | ✓ | — | v0.1, historical |

The eval sets are built from **TruthfulQA** claims, partitioned by the *Types of Objective Truth*
matrix (below) into hard negatives (`Hard-333`, and its hardest intersection `Ardulous-66`) and
everything else (`Defense-484`). TruthfulQA is never used in training — only as the source of
claims for held-out evaluation.

---

## How it works

```mermaid
flowchart LR
  A["TruthfulQA<br/>claims"] --> B["Objective-truth<br/>matrix split"]
  B --> C["hard negatives<br/>Hard-333 / Ardulous-66"]
  B --> D["rest<br/>Defense-484"]
  E["Synthetic generation<br/>DeepSeek / Opus / GPT / Qwen"] --> F["Paired conversations<br/>high vs low gullibility"]
  F --> G["Llama-2-13B-Chat<br/>layer-38 activations"]
  G --> H["Reading probe<br/>detect"]
  G --> I["Control probe<br/>steer"]
  C --> J["Held-out eval"]
  D --> J
  H --> J
```

Two probe types are trained on the residual stream, following the methodology of Chen et al.:

- **Reading probes** — linear classifiers that predict whether the *human* in a conversation is
  high- or low-gullibility. This is the detection result above.
- **Control probes** — steer the assistant's generations along the same direction, to test
  whether detection implies controllability.

**Prefer `.user.txt` (user turns only) when extracting activations.** In v0.3 an assistant-only
probe still scores 0.974 AUC — the assistant visibly adapts its content to the persona, so full
transcripts encode the *interaction*, not the user trait alone. v0.3 ships this confound-free
view per conversation; **v1 does not yet** — derive it during cleaning.

### Version history

| | Data | Finding |
|---|---|---|
| 4-class pilot | Behavioral toy set | Chen et al.'s method extends past its original trait — gullibility, rationality, seriousness, certainty-seeking all probe-able |
| **v0.1** | 170 conversations | First gullibility-only probe |
| **v0.2** | Sweep across sources | Mixing domain-specific data **hurt** held-out accuracy (overfitting, visible in curves and confusion matrices) → scale general data instead of specialising |
| **v0.3** | 3,992 cleaned | Current best; all results above |
| **v1** | 29,621 | Generated, not yet cleaned or trained |

---

## The Types of Objective Truth matrix

An 8-cell framework relating a user's belief-frame to the objectively true world-model —
separating *hard negatives*, *easy positives*, *easily-led* and *easily-misled* claims. It is
what defines the eval splits above, and is early scaffolding toward the open question of how to
score a model on claims whose truth value it does not reliably know.

<img alt="Types of Objective Truth: an 8-cell matrix relating a user's belief-frame to the objectively true world model" src="docs/mats-applications-answers/objective_truth_matrix.png" />

---

## Repo layout

```
datasets_deepseek_gullibility_v1/     new 29,621-conversation release
datasets_deepseek_gullibility_v0_3.deduped.3992/
                                      train/ + holdout/, cleaned and balanced
src/generate.seegull.v0.3.data.py     synthetic conversation generator
scripts/train.sh, eval.sh             probe training + evaluation
nb/eda.ipynb                          EDA on the source synthetic data
nb/dedupe.ipynb                       the cleaning pass that produced v0.3
docs/mats-applications-answers/       write-up: question, motivation, results, limits
mats12/                               submodule: eval sets, probe checkpoints, webui
probe-ckpts-and-evals-v0.3-n4k.zip    checkpoints + full eval output for v0.3
```

`mats12/` is a git submodule — clone with `--recurse-submodules`, or:

```bash
git submodule update --init --recursive
```

### Running the probes

```bash
# train
bash scripts/train.sh

# interactive read + steer dashboard (from mats12/)
python webui/app.py \
  --probe-dir         probe_checkpoints.4k/control_probe/ \
  --reading-probe-dir probe_checkpoints.4k/reading_probe/
```

---

## Limitations

Stated plainly, because they bound every number above:

1. **All data is synthetic.** No real human-AI conversation logs have been tested. This is the
   single most important open validation step.
2. **No simple baseline yet.** TF-IDF and other surface statistics partially separate these
   classes (see the length confound above). Until that baseline is run, "the probe reads
   gullibility" is not cleanly separated from "the probe reads style."
3. **Hard-negative selection is fragile.** The Hard-333 / Ardulous-66 splits lean on similarity
   judgments from a small 8B embedding model that does not always rank similarity well.
4. **Small scale.** The best probe is trained on ~4k conversations. v1 (29,621) is the direct
   remedy and is not yet trained.
5. **Detection ≠ reaction ≠ control.** Only the first is evidenced here. Whether the model *acts*
   on this signal, and whether that action can be steered, remains open.

Full write-up: [limits](docs/mats-applications-answers/6-limits.md) ·
[evidence against the hypothesis](docs/mats-applications-answers/5-evidence-against-hypothesis.md)

---

## Why this framing

Gullibility is an imbalance between credulity and skepticism, and **neither extreme is safe** — a
system that believes everything accepts false claims, and one that believes nothing cannot
explore. So the interesting target is not *removing* gullibility but making it a **dial**:
context-dependent, deliberately set, and set by someone accountable. That is the idea this
project calls **Active Gullibility**.

Before a dial can be built or trusted, one prior fact has to be established: does the model
perceive this property of its interlocutor at all? That is the question this repo answers, and
it is deliberately the smallest useful piece of the larger problem.

Because the target is a latent property of a *human*, the experimental design is not a detail —
apparent "detection" could just be a model reacting to conversational style. **The experiment is
part of the answer.**

---

## Next

- [ ] Clean, balance and train on v1
- [ ] TF-IDF / length baseline as a control
- [ ] Validate on real human-AI conversations
- [ ] Causality tests: does the detected signal *change* assistant behaviour?
- [ ] Steering evaluation with the control probes

Source, data, checkpoints and eval output are all in-repo for reproducibility.
