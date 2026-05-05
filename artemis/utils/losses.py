import torch
import torch.nn as nn
import torch.nn.functional as F

def empirical_entropy_from_labels(t_idx: torch.Tensor, num_treatments: int, eps: float = 1e-8) -> torch.Tensor:
    counts = torch.bincount(t_idx.view(-1).long(), minlength=num_treatments).float()
    probs = counts / counts.sum().clamp_min(1.0)
    return -(probs * torch.log(probs + eps)).sum()

def treatment_log_prob_mean(
        classifier: nn.Module,
        z: torch.Tensor,
        t_idx: torch.Tensor,
        num_treatments: int
) -> torch.Tensor:
    logits = classifier(z)
    if num_treatments == 2:
        t_float = t_idx.view(-1, 1).float()
        return (t_float * F.logsigmoid(logits) + (1.0 - t_float) * F.logsigmoid(-logits)).mean()
    log_probs = F.log_softmax(logits, dim=1)
    return log_probs.gather(1, t_idx.view(-1, 1).long()).mean()

def treatment_classifier_loss(
        classifier: nn.Module,
        z: torch.Tensor,
        t_idx: torch.Tensor,
        num_treatments: int
) -> torch.Tensor:
    logits = classifier(z)
    if num_treatments == 2:
        return F.binary_cross_entropy_with_logits(logits, t_idx.view(-1, 1).float())
    return F.cross_entropy(logits, t_idx.view(-1).long())

def variational_mi_lower_bound(
        classifier: nn.Module,
        z: torch.Tensor,
        t_idx: torch.Tensor,
        num_treatments: int
) -> torch.Tensor:
    return (
        empirical_entropy_from_labels(t_idx.detach(), num_treatments)
        + treatment_log_prob_mean(classifier, z, t_idx, num_treatments)
    )

def contrastive_loss(z1, z2, label, margin=1.0):
    dist_sq = torch.sum((z1 - z2) ** 2, dim=1)
    loss_sim = label.float() * dist_sq if label.dtype != torch.float32 else label * dist_sq
    loss_dissim = (1.0 - label.float()) * torch.clamp(margin - torch.sqrt(dist_sq + 1e-8), min=0.0) ** 2
    return torch.mean(loss_sim + loss_dissim) / 2.0
