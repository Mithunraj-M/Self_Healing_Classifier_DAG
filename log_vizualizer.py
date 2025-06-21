# log_visualizer.py

import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

LOG_DIR = "logs"
CONFIDENCE_LOG = os.path.join(LOG_DIR, "inference.log")
FALLBACK_LOG = os.path.join(LOG_DIR, "fallback.log")
CHECK_LOG = os.path.join(LOG_DIR, "confidence_check.log")

sns.set(style="whitegrid")

# ---------- Utility Parsers ----------

def parse_inference_log():
    confidences = []
    labels = []
    if not os.path.exists(CONFIDENCE_LOG):
        return confidences, labels

    with open(CONFIDENCE_LOG, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"Prediction: (\w+) \| Confidence: ([0-9.]+)", line)
            if match:
                labels.append(match.group(1))
                confidences.append(float(match.group(2)))
    return confidences, labels

def parse_fallback_log():
    count = 0
    if not os.path.exists(FALLBACK_LOG):
        return 0
    with open(FALLBACK_LOG, "r", encoding="utf-8") as f:
        for line in f:
            if "[FALLBACK]" in line:
                count += 1
    return count

def parse_check_log():
    accepted = 0
    fallback = 0
    if not os.path.exists(CHECK_LOG):
        return accepted, fallback
    with open(CHECK_LOG, "r", encoding="utf-8") as f:
        for line in f:
            if "fallback" in line.lower():
                fallback += 1
            elif "accept" in line.lower():
                accepted += 1
    return accepted, fallback

# ---------- Plotting ----------

def plot_confidence_histogram(confidences):
    plt.figure(figsize=(8, 5))
    sns.histplot(confidences, bins=10, kde=True, color="cornflowerblue")
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_label_confidence(confidences, labels):
    by_label = defaultdict(list)
    for c, l in zip(confidences, labels):
        by_label[l].append(c)

    means = {k: sum(v)/len(v) for k, v in by_label.items()}
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(means.keys()), y=list(means.values()), palette="muted")
    plt.title("Mean Confidence by Label")
    plt.ylim(0, 1)
    plt.ylabel("Average Confidence")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_fallback_frequency(accepts, fallbacks):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Accepted", "Fallbacks"], y=[accepts, fallbacks], palette=["green", "red"])
    plt.title("Prediction Outcome Frequency")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# ---------- Main ----------
if __name__ == "__main__":
    confidences, labels = parse_inference_log()
    accepted, fallback = parse_check_log()

    print(f"Parsed {len(confidences)} predictions")
    print(f"Accepted: {accepted} | Fallbacks: {fallback}")

    if confidences:
        plot_confidence_histogram(confidences)
        plot_label_confidence(confidences, labels)
        plot_fallback_frequency(accepted, fallback)
    else:
        print("No confidence data to plot yet. Run the app with more inputs first.")
