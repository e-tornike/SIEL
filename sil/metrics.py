import torch
import numpy as np
from torchmetrics import F1Score, Precision, Recall
import math


def compute_metrics(pred):
    f1 = F1Score(task="binary")
    r = Recall(task="binary")
    p = Precision(task="binary")
    f1micro = F1Score(average="micro", task="multiclass", num_classes=2)
    rmicro = Recall(average="micro", task="multiclass", num_classes=2)
    pmicro = Precision(average="micro", task="multiclass", num_classes=2)

    metrics = [f1, f1micro, r, rmicro, p, pmicro]

    scores = {}
    for m in metrics:
        preds, labels = pred
        score = m(preds=torch.Tensor(np.argmax(preds, axis=1)), target=torch.Tensor(labels))
        scores[str(m)] = float(score)
    return scores


def compute_fine_grained_metrics(df, attribute_columns, target_col, pred_col):
    results = {}

    for acol in attribute_columns:
        for name,group in df.groupby(by=acol):
            group_preds = torch.tensor(group[pred_col].tolist())
            group_labels = torch.tensor(group[target_col].tolist())
            pred_scores = compute_metrics([group_preds, group_labels])
            results[acol+":"+name] = pred_scores

    return results


def get_score(hits, labels, method="cg", return_probs=True):
    assert method in ["cg", "rdcg", "dcg", "dcg_burges"]

    true_score = 0.0
    false_score = 0.0
    for i,hit in enumerate(hits):
        hcorpus_id = hit['corpus_id']
        label = labels[hcorpus_id]

        score = hit['score']
        if method == "cg":
            score = score
        elif method == "rdcg":
            score = (score * (1/(i+1)))
        elif method == "dcg":
            score = (score/(math.log(i+2, 2)))
        elif method == "dcg_burges":
            score = (((2**score)-1)/(math.log(i+2, 2)))

        if label in [1, '1']:
            true_score += score
        else:
            false_score += score
    
    if return_probs:
        total_score = true_score + false_score
        true_prob = true_score / total_score
        false_prob = false_score / total_score
        return true_prob, false_prob
    else:
        return true_score, false_score
