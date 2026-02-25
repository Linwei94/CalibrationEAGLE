'''
Metrics to measure calibration of a trained deep neural network.

References:
[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks.
    arXiv preprint arXiv:1706.04599, 2017.
'''

import argparse
import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

#for plotting
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

# Some keys used for the following dictionaries
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + \
            (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                float(bin_dict[binn][COUNT])
    return bin_dict


def expected_calibration_error(confs, preds, labels, num_bins=10):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * \
            abs(bin_accuracy - bin_confidence)
    return ece


def maximum_calibration_error(confs, preds, labels, num_bins=10):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    ce = []
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        ce.append(abs(bin_accuracy - bin_confidence))
    return max(ce)


def average_calibration_error(confs, preds, labels, num_bins=10):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    non_empty_bins = 0
    ace = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        if bin_count > 0:
            non_empty_bins += 1
        ace += abs(bin_accuracy - bin_confidence)
    return ace / float(non_empty_bins)


def l2_error(confs, preds, labels, num_bins=15):
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    l2_sum = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        l2_sum += (float(bin_count) / num_samples) * \
               (bin_accuracy - bin_confidence)**2
        l2_error = math.sqrt(l2_sum)
    return l2_error


# Calibration error scores in the form of loss metrics
class ECELoss(nn.Module):
    '''
    Compute ECE (Expected Calibration Error)
    '''
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class AdaptiveECELoss(nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class ClasswiseECELoss(nn.Module):
    '''
    Compute Classwise ECE
    '''
    def __init__(self, n_bins=15):
        super(ClasswiseECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).item())
        softmaxes = F.softmax(logits, dim=1)
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=logits.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce




class Calibration:
    def __init__(self, method="histogram", n_bins=15):
        """
        method: "histogram" or "platt"
        n_bins: number of bins for histogram binning
        """
        self.method = method
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_acc = None
        self.platt_model = None

    # -------------------------------------------------
    # Fit
    # -------------------------------------------------
    def fit(self, pred_label, gt_label, pred_confidence):
        pred_label = np.array(pred_label)
        gt_label = np.array(gt_label)
        pred_confidence = np.array(pred_confidence)

        correct = (pred_label == gt_label).astype(int)

        if self.method == "histogram":
            self._fit_histogram(pred_confidence, correct)

        elif self.method == "platt":
            self._fit_platt(pred_confidence, correct)

        else:
            raise ValueError("method must be 'histogram' or 'platt'")

    # -------------------------------------------------
    # Histogram Binning
    # -------------------------------------------------
    def _fit_histogram(self, confidences, correct):
        self.bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        self.bin_acc = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            in_bin = (confidences >= self.bin_edges[i]) & \
                     (confidences < self.bin_edges[i + 1])

            if np.sum(in_bin) > 0:
                self.bin_acc[i] = np.mean(correct[in_bin])
            else:
                self.bin_acc[i] = 0.0

    def _transform_histogram(self, confidences):
        calibrated = np.zeros_like(confidences)

        for i in range(self.n_bins):
            in_bin = (confidences >= self.bin_edges[i]) & \
                     (confidences < self.bin_edges[i + 1])
            calibrated[in_bin] = self.bin_acc[i]

        return calibrated

    # -------------------------------------------------
    # Platt Scaling
    # -------------------------------------------------
    def _fit_platt(self, confidences, correct):
        # convert confidence -> logit
        eps = 1e-6
        confidences = np.clip(confidences, eps, 1 - eps)
        logits = np.log(confidences / (1 - confidences)).reshape(-1, 1)

        self.platt_model = LogisticRegression()
        self.platt_model.fit(logits, correct)

    def _transform_platt(self, confidences):
        eps = 1e-6
        confidences = np.clip(confidences, eps, 1 - eps)
        logits = np.log(confidences / (1 - confidences)).reshape(-1, 1)

        calibrated = self.platt_model.predict_proba(logits)[:, 1]
        return calibrated

    # -------------------------------------------------
    # Public transform
    # -------------------------------------------------
    def transform(self, pred_confidence):
        pred_confidence = np.array(pred_confidence)

        if self.method == "histogram":
            return self._transform_histogram(pred_confidence)

        elif self.method == "platt":
            return self._transform_platt(pred_confidence)

    # -------------------------------------------------
    # ECE
    # -------------------------------------------------
    def compute_ece(self, pred_label, gt_label, pred_confidence):
        pred_label = np.array(pred_label)
        gt_label = np.array(gt_label)
        pred_confidence = np.array(pred_confidence)

        correct = (pred_label == gt_label).astype(int)

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece = 0.0
        n = len(pred_confidence)

        for i in range(self.n_bins):
            in_bin = (pred_confidence >= bin_edges[i]) & \
                     (pred_confidence < bin_edges[i + 1])

            if np.sum(in_bin) > 0:
                bin_acc = np.mean(correct[in_bin])
                bin_conf = np.mean(pred_confidence[in_bin])
                ece += np.abs(bin_acc - bin_conf) * np.sum(in_bin) / n

        return ece


def _compute_bin_stats(confs, correctness, n_bins):
    """Compute bin_acc, bin_conf, bin_counts, bin_error_signed, ece for given confidences."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confs, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    bin_acc = []
    bin_conf = []
    bin_counts = []

    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) > 0:
            bin_acc.append(np.mean(correctness[mask]))
            bin_conf.append(np.mean(confs[mask]))
            bin_counts.append(np.sum(mask))

    bin_acc = np.array(bin_acc)
    bin_conf = np.array(bin_conf)
    bin_counts = np.array(bin_counts)

    bin_error_signed = bin_acc - bin_conf
    bin_error_abs = np.abs(bin_error_signed)
    total_samples = len(confs)
    ece = np.sum((bin_counts / total_samples) * bin_error_abs)

    return bin_acc, bin_conf, bin_counts, bin_error_signed, ece


def plot_reliability_with_error(
    pred_labels,
    gt_labels,
    pred_confs,
    calibrated_confs,
    n_bins=10,
    position_id=None,
    save_path=None
):
    # ---- convert to numpy ----
    pred_labels = np.array(pred_labels)
    gt_labels = np.array(gt_labels)
    pred_confs = np.array(pred_confs)
    calibrated_confs = np.array(calibrated_confs)

    correctness = (pred_labels == gt_labels).astype(int)

    # --------- pre-calibration stats ----------
    pre_bin_acc, pre_bin_conf, _, pre_bin_error_signed, pre_ece = _compute_bin_stats(
        pred_confs, correctness, n_bins
    )

    # --------- post-calibration stats ----------
    post_bin_acc, post_bin_conf, _, post_bin_error_signed, post_ece = _compute_bin_stats(
        calibrated_confs, correctness, n_bins
    )

    # --------- plotting: 2x2 layout ----------
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # =====================
    # Row 0: Pre-calibration
    # =====================
    # Reliability Diagram (pre)
    ax_pre_rel = axes[0, 0]
    ax_pre_rel.plot(pre_bin_conf, pre_bin_acc, marker='o')
    ax_pre_rel.plot([0, 1], [0, 1], linestyle="--", color="gray")
    for i in range(len(pre_bin_conf)):
        ax_pre_rel.text(
            pre_bin_conf[i], pre_bin_acc[i],
            f"{pre_bin_error_signed[i]:+.3f}",
            fontsize=9, ha='left', va='bottom'
        )
    ax_pre_rel.set_xlabel("Mean Predicted Confidence")
    ax_pre_rel.set_ylabel("Empirical Accuracy")
    ax_pre_rel.set_xlim(0, 1)
    ax_pre_rel.set_ylim(0, 1)
    ax_pre_rel.set_title(f"Pre-calibration Reliability (ECE={pre_ece:.4f}, Tree Depth={position_id})")

    # Confidence Density (pre)
    ax_pre_den = axes[0, 1]
    sns.kdeplot(pred_confs, fill=True, ax=ax_pre_den)
    ax_pre_den.set_xlim(0, 1)
    ax_pre_den.set_xlabel("Prediction Confidence")
    ax_pre_den.set_ylabel("Density")
    ax_pre_den.set_title("Pre-calibration Confidence Density")

    # =====================
    # Row 1: Post-calibration
    # =====================
    # Reliability Diagram (post)
    ax_post_rel = axes[1, 0]
    ax_post_rel.plot(post_bin_conf, post_bin_acc, marker='o')
    ax_post_rel.plot([0, 1], [0, 1], linestyle="--", color="gray")
    for i in range(len(post_bin_conf)):
        ax_post_rel.text(
            post_bin_conf[i], post_bin_acc[i],
            f"{post_bin_error_signed[i]:+.3f}",
            fontsize=9, ha='left', va='bottom'
        )
    ax_post_rel.set_xlabel("Mean Predicted Confidence")
    ax_post_rel.set_ylabel("Empirical Accuracy")
    ax_post_rel.set_xlim(0, 1)
    ax_post_rel.set_ylim(0, 1)
    ax_post_rel.set_title(f"Post-calibration Reliability (ECE={post_ece:.4f}, Tree Depth={position_id})")

    print(f"Calibration: {pre_ece:.4f} --> {post_ece:.4f}")

    # Confidence Density (post)
    ax_post_den = axes[1, 1]
    sns.kdeplot(calibrated_confs, fill=True, ax=ax_post_den)
    ax_post_den.set_xlim(0, 1)
    ax_post_den.set_xlabel("Prediction Confidence")
    ax_post_den.set_ylabel("Density")
    ax_post_den.set_title("Post-calibration Confidence Density")

    plt.tight_layout()


    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"reliability_and_density_{position_id}.png"), dpi=300, bbox_inches="tight")

    plt.show()
    
    
def main(args):
    K = 0 # first K tokens in each verification batch are excluded
    N_BINS = 15
    POSITION_ID = args.position_id
    
    def read_calibration_stats(calibration_stats_path):
        calibration_stats = torch.load(calibration_stats_path)
        draft_tokens_list = []
        draft_probs_list = []
        target_tokens_list = []
        mask_list = []
        for i in range(len(calibration_stats)):
            tree_position_ids = calibration_stats[i]["tree_position_ids"]
            mask = torch.cat([tree_position_ids[j][K+1:] for j in range(len(tree_position_ids))])==POSITION_ID
            mask_list.append(mask)
            draft_tokens = calibration_stats[i]["draft_tokens"]
            draft_tokens = torch.cat([draft_tokens[j][K+1:] for j in range(len(draft_tokens))])[mask].flatten()
            target_tokens = calibration_stats[i]["target_tokens"]
            target_tokens = torch.cat([target_tokens[j][K+1:] for j in range(len(target_tokens))])[mask].flatten()
            draft_probs = calibration_stats[i]["draft_probs"]
            draft_probs = torch.cat([draft_probs[j][K:] for j in range(len(draft_probs))])[mask].flatten()
            draft_tokens_list.append(draft_tokens)
            draft_probs_list.append(draft_probs)
            target_tokens_list.append(target_tokens)
            
        draft_tokens_list = torch.cat(draft_tokens_list).tolist()
        draft_probs_list = torch.cat(draft_probs_list).tolist()
        target_tokens_list = torch.cat(target_tokens_list).tolist()
        return draft_tokens_list, draft_probs_list, target_tokens_list
    
    #warmup data
    warmup_calibration_stats_path = f"{args.calibration_stat_file_root}/calibration_stats_warmup.pth"
    warmup_draft_tokens_list, warmup_draft_probs_list, warmup_target_tokens_list = read_calibration_stats(warmup_calibration_stats_path)
    #evaluation data
    eval_calibration_stats_path = f"{args.calibration_stat_file_root}/calibration_stats_eval.pth"
    eval_draft_tokens_list, eval_draft_probs_list, eval_target_tokens_list = read_calibration_stats(eval_calibration_stats_path)
    #ood data
    ood_calibration_stats_path = f"{args.ood_calibration_stat_file_root}/calibration_stats_eval.pth"
    ood_draft_tokens_list, ood_draft_probs_list, ood_target_tokens_list = read_calibration_stats(ood_calibration_stats_path)
    
    #calibration using first 20% eval data
    N = int(0.2 * len(eval_draft_probs_list))
    calibrator = Calibration(method="platt")
    calibrator.fit(eval_draft_tokens_list[:N], eval_target_tokens_list[:N], eval_draft_probs_list[:N])
    calibrated_probs = calibrator.transform(eval_draft_probs_list[N:])
    plot_reliability_with_error(eval_draft_tokens_list[N:], eval_target_tokens_list[N:], eval_draft_probs_list[N:], calibrated_probs, n_bins=N_BINS, position_id=POSITION_ID, save_path=args.calibration_stat_file_root)
    
    #ood data
    calibrated_probs = calibrator.transform(ood_draft_probs_list)
    plot_reliability_with_error(ood_draft_tokens_list, ood_target_tokens_list, ood_draft_probs_list, calibrated_probs, n_bins=N_BINS, position_id=POSITION_ID, save_path=f"{args.calibration_stat_file_root}/ood_calibration")

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration_stat_file_root", default="eagle/calibration/mt_bench/llama38b2_40-temperature-0.0/EAGLE3")
    parser.add_argument("--ood_calibration_stat_file_root", default="eagle/calibration/gsm8k/llama38b2_40-temperature-0.0/EAGLE3")
    args = parser.parse_args()
    
    for pid in range(1, 7):
        args.position_id = pid
        main(args)