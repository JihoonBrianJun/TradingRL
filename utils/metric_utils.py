import torch

def compute_predictor_metrics(pred, target, value_threshold, strong_threshold):
    metrics = dict()
    metrics["correct"] = ((pred*target)>0).sum().item()

    metrics["rec_tgt"] = (target>=value_threshold).to(torch.long).sum().item() + (target<=-value_threshold).to(torch.long).sum().item()
    rec_correct_pos = ((target>=value_threshold).to(torch.long) * (pred>0).to(torch.long)).sum().item()
    rec_correct_neg = ((target<=-value_threshold).to(torch.long) * (pred<0).to(torch.long)).sum().item()
    metrics["rec_correct"] = rec_correct_pos + rec_correct_neg

    metrics["strong_prec_tgt"] = (pred>=strong_threshold).to(torch.long).sum().item() + (pred<=-strong_threshold).to(torch.long).sum().item()
    strong_prec_correct_pos = ((pred>=strong_threshold).to(torch.long) * (target>0).to(torch.long)).sum().item()
    strong_prec_correct_neg = ((pred<=-strong_threshold).to(torch.long) * (target<0).to(torch.long)).sum().item()
    metrics["strong_prec_correct"] = strong_prec_correct_pos+ strong_prec_correct_neg

    return metrics