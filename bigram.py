import random
from pathlib import Path
from collections import Counter, defaultdict

random.seed(42)

# ---------------------------------------------------------------
# 1. Load and prepare the data
# ---------------------------------------------------------------
words = Path('names.txt').read_text(encoding='utf-8').splitlines()
assert len(words) == 32033, f"expected 32033 names, got {len(words)}"

padded_words = [f".{word}." for word in words]

# ---------------------------------------------------------------
# 2. Vocabulary: every token the model knows
# ---------------------------------------------------------------
vocab = sorted(set(''.join(padded_words)))

# ---------------------------------------------------------------
# 3. "Training": count every consecutive pair
# ---------------------------------------------------------------
bigram_counts = Counter()
for word in padded_words:
    for ch1, ch2 in zip(word, word[1:]):
        bigram_counts[(ch1, ch2)] += 1

assert sum(bigram_counts.values()) == 228146, "pair count is off, check padding"

transitions = defaultdict(Counter)
for (ch1, ch2), count in bigram_counts.items():
    transitions[ch1][ch2] = count


# ---------------------------------------------------------------
# 4. The model: counts -> probability distribution over next char
# ---------------------------------------------------------------
def next_char_probs(ch):
    if ch not in transitions:
        raise KeyError(f"'{ch}' is not in the vocabulary")

    followers = transitions[ch]
    total = sum(followers.values())
    probs = {next_ch: count / total for next_ch, count in followers.items()}

    assert abs(sum(probs.values()) - 1) < 1e-9, "probabilities must sum to 1"
    return probs


# ---------------------------------------------------------------
# 5. Inference: two decoding strategies
# ---------------------------------------------------------------
def sample_name(max_len=50):
    """Pick the next char at random, weighted by probability."""
    out, curr = [], '.'
    while len(out) < max_len:
        probs = next_char_probs(curr)
        curr = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
        if curr == '.':
            break
        out.append(curr)
    return ''.join(out)


def greedy_name(max_len=50):
    """Always pick the single most likely next char."""
    out, curr = [], '.'
    while len(out) < max_len:
        probs = next_char_probs(curr)
        curr = max(probs, key=lambda ch: probs[ch])
        if curr == '.':
            break
        out.append(curr)
    return ''.join(out)


if __name__ == "__main__":
    print(f"Vocab ({len(vocab)}): {''.join(vocab)}\n")

    print("Top 5 after 'a':")
    for ch, p in Counter(next_char_probs('a')).most_common(5):
        print(f"  {ch}  {p:.1%}")

    print("\nSampled names:")
    for _ in range(10):
        print(" ", sample_name())

    print("\nGreedy names:")
    for _ in range(3):
        print(" ", repr(greedy_name()))
