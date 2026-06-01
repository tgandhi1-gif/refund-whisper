"""The Refund Whisper - predict refund intent from support chat text."""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(13)
refund_words = ["cancel", "refund", "broken", "useless", "disappointed", "charge", "money back", "waited", "still not"]
happy_words = ["thanks", "great", "love", "helpful", "quick", "works", "appreciate", "perfect", "solved"]
neutral_words = ["order", "item", "team", "support", "email", "today", "account", "package", "update", "hello"]

def make(words, other, n, label):
    rows = []
    for _ in range(n):
        # about 30% of tickets are genuinely mixed in tone; the rest lean one way
        # but still borrow a few words from the other side - real chats are never clean
        if rng.random() < 0.30:
            pool = list(words) + list(other)
            msg = list(rng.choice(pool, rng.integers(6, 12))) + list(rng.choice(neutral_words, 3))
        else:
            msg = (list(rng.choice(words, rng.integers(5, 9)))
                   + list(rng.choice(other, rng.integers(2, 5)))
                   + list(rng.choice(neutral_words, 3)))
        rng.shuffle(msg)
        rows.append((" ".join(msg), label))
    return rows

data = make(refund_words, happy_words, 1500, 1) + make(happy_words, refund_words, 1500, 0)
rng.shuffle(data)
texts = [t for t, _ in data]
y = np.array([l for _, l in data])

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X = vec.fit_transform(texts)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
print(f"Refund intent model AUC: {auc:.3f}")

coef = clf.coef_[0]
names = np.array(vec.get_feature_names_out())
top = np.argsort(coef)[::-1][:12]
print("Strongest refund whispers:", list(names[top][:6]))

os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.barh(names[top][::-1], coef[top][::-1], color="#ff6a3d")
plt.title("Words that whisper a refund is coming")
plt.tight_layout()
plt.savefig("outputs/refund_whisper.png", dpi=120)
print("Saved outputs/refund_whisper.png")
