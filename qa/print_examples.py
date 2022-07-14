"""Score generated predictions."""

import argparse
import collections
import re
import string
import tqdm
import json
import numpy as np


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, answer):
    """Compute F1 score between prediction tokens and ground truth tokens."""
    prediction_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    common = (collections.Counter(prediction_tokens) &
              collections.Counter(answer_tokens))
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(answer_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_answers(metric_fn, prediction, answer_set):
    """Return the maximum score between any (prediction, answer) pair."""
    max_score = -float("inf")
    for answer in answer_set:
        score = metric_fn(prediction, answer)
        max_score = max(max_score, score)
    return max_score


def load_predictions(filename, threshold, top_docs=100):
    print("Reading dataset")
    with open(filename, "r") as f:
        raw_data = json.load(f)

    dataset = []
    for entry in tqdm.tqdm(raw_data, "processing data..."):
        # Get answers.
        question = entry["question"]
        answers = entry["gold_answers"]

        # Get predictions.
        predictions = None
        for p in entry["predictions"]:
            if p["top_k"] == top_docs:
                predictions = p["predictions"]
                break
        if predictions is None:
            raise RuntimeError(f"Could not find entry corresponding to top_k={top_docs}.")

        # Score predictions by F1, and sort by joint doc * span confidence score.
        deduped = set()
        prediction_set = []
        max_f1 = 0

        for p in predictions:
            if normalize_answer(p["text"]) in deduped:
                continue
            score = p["score"] + p["relevance_score"]
            if -score > threshold:
                continue
            f1 = metric_max_over_answers(f1_score, p["text"], answers)
            prediction_set.append(p["text"])
            max_f1 = max(max_f1, f1)
            deduped.add(normalize_answer(p["text"]))

        dataset.append((prediction_set, question, answers, max_f1))

    return dataset


def main(args):
    dataset = load_predictions(args.input_file, args.threshold, args.top_docs)
    np.random.shuffle(dataset)

    with open("out", "w") as f:
        for i in range(len(dataset)):
            p_set, q, a, f1 = dataset[i]
            f.write("=" * 10 + "\n")
            f.write("%s" % q + "\n")
            f.write("%s" % a + "\n")
            f.write("%s" % f1 + "\n")
            f.write("Predictions:" + "\n")
            for p in p_set:
                f.write("\t%s" % p + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="results/predictions.json")
    parser.add_argument("--threshold", type=float, default=-5.201)
    parser.add_argument("--top_docs", type=int, default=100)
    args = parser.parse_args()
    main(args)
