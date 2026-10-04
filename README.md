# 01 · Bigram Language Model (from scratch, no ML libraries)

The simplest possible language model, built in plain Python to understand what an LLM actually is before moving on to neural networks and transformers.

## What it does

Trains on 32,033 first names and generates new ones. It predicts each next character by looking only at the previous one: a bigram model.

```bash
python3 bigram.py
```

## How it works

| Step | What I did | Equivalent in a real LLM |
|---|---|---|
| Data | Load 32,033 names, wrap each in `.` as a start/end marker | Training corpus + BOS/EOS special tokens |
| Vocabulary | 27 tokens: `a–z` plus `.` | Tokenizer vocabulary (50k–200k tokens) |
| Training | Count every consecutive character pair (228,146 pairs) | Gradient descent over billions of weights |
| Model | Normalise counts into a probability distribution over the next char | Final softmax layer |
| Inference | Generate in a loop until the end token, capped by `max_len` | Decoding with `max_tokens` |

## What I learned

**An LLM is a next-token probability function called in a loop.** My bigram and GPT share the same structure; GPT just predicts far better because it sees far more context.

**The end token is what makes a model stop.** Without the `.` markers the model can't learn how names begin or end, and generation would never terminate.

**Greedy decoding collapses.** Always picking the most likely next character produces `.` → `a` → `.`, the name "a", every single time. Sampling from the distribution gives variety, which is why chat models sample and expose a `temperature` setting.

**The model's weakness is context.** With only one character of memory it can't tell where it is in a name, so outputs are a mix of plausible and nonsense. Fixing this is exactly what attention does.

## A bug I caught

My first version built the padded words but looped over the unpadded list. Nothing crashed, and the output looked reasonable. A checksum assertion exposed it: I got 164,080 pairs instead of 228,146, a gap of exactly 64,066 = 2 × 32,033, one missing start and end pair per name. That exact multiple pointed straight to the cause.

The lesson: ML bugs are usually silent. I now treat data assertions (sizes, totals, probabilities summing to 1) the way I treat unit tests.

## Sample output

```
$ python bigram.py
Vocab (27): .abcdefghijklmnopqrstuvwxyz

Top 5 after 'a':
  .  19.6%
  n  16.0%
  r  9.6%
  l  7.5%
  h  6.9%

Sampled names:
  n
  styaa
  his
  e
  brinivey
  aslialyeancanininn
  ste
  are
  marren
  ke

Greedy names:
  'a'
  'a'
  'a'
```

Some samples look like real names ("marren", "his"), while "aslialyeancanininn" shows the one-character memory problem: the model has no idea how long the name already is. Greedy decoding returns "a" every time, because `.` → `a` and `a` → `.` are each the single most likely step.

## Next

Measuring model quality with a single number (negative log-likelihood loss), then replacing the count table with a neural network trained by gradient descent.
