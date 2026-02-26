import torch

# typing
from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression

TOPK = 10  # topk for sparse tree


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

class node:
    def __init__(self,parent=None,value=None,dict_key=None):
        self.parent=parent
        self.value=value
        if parent:
            self.depth=parent.depth+1
            parent.children.append(self)
        else:
            self.depth=0
        self.children=[]
        self.dict_key=dict_key
    def is_leaf(self):
        return len(self.children)==0

    def all_index(self):
        if not self.parent.parent:
            return [self.index]
        else:
            return self.parent.all_index()+[self.index]



class Tree:
    def __init__(self,tree_list):
        sorted_tree_list = sorted(tree_list, key=lambda x: (len(x), x))
        self.root=node()
        self.node_dic={}
        for tree_node in sorted_tree_list:
            cur_value=tree_node[-1]
            if len(tree_node)==1:
                cur_node=node(parent=self.root,value=cur_value,dict_key=tuple(tree_node))
            else:
                cur_parent=self.node_dic[tuple(tree_node[:-1])]
                cur_node = node(parent=cur_parent, value=cur_value,dict_key=tuple(tree_node))
            self.node_dic[tuple(tree_node)] = cur_node
        self.indexnode()

    def max_depth(self):
        return max([item.depth for item in self.node_dic.values()])

    def num_node_wchild(self):
        num_c=0
        for item in self.node_dic.values():
            if not item.is_leaf():
                num_c+=1
        return num_c

    def get_node_wchild(self):
        ns=[]
        for item in self.node_dic.values():
            if not item.is_leaf():
                ns.append(item)
        return ns

    def indexnode(self):
        cur_index=0
        for key in self.node_dic:
            cur_node=self.node_dic[key]
            if not cur_node.is_leaf():
                cur_node.index=cur_index
                cur_index+=1




def generate_tree_buffers(tree_choices, device="cuda"):
    tree=Tree(tree_choices)
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = tree.num_node_wchild()


    max_depth=tree.max_depth()
    nodes_wc=tree.get_node_wchild()

    depth_counts=[0 for _ in range(max_depth-1)]
    for x in nodes_wc:
        depth_counts[x.depth-1]+=1
    depth_counts_sum = [sum(depth_counts[:i + 1]) for i in range(len(depth_counts))]


    tree_attn_mask = torch.eye(tree_len, tree_len)

    for id,x in enumerate(nodes_wc):
        tree_attn_mask[id,x.all_index()]=1




    tree_attn_mask_list0=[tree_attn_mask[:ml,:ml] for ml in depth_counts_sum]
    tree_attn_mask_list=[]
    for id,x in enumerate(tree_attn_mask_list0):
        x=x[-depth_counts[id]:]
        tree_attn_mask_list.append(x)



    tree_indices_list = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]
    repeat_nums=[[] for _ in depth_counts]
    start = 0
    bias = 0
    for i in range(len(depth_counts)):
        bias = 0
        repeat_j=0
        for j in range(depth_counts[i]):
            cur_node = nodes_wc[start + j]
            cur_parent = cur_node.parent

            if j != 0:
                if cur_parent != parent:
                    bias += 1
                    parent = cur_parent
                    repeat_nums[i].append(j-repeat_j)
                    repeat_j=j
            else:
                parent = cur_parent
            tree_indices_list[i][j] = cur_node.value + TOPK * (bias)
        repeat_nums[i].append(j - repeat_j+1)
        start += depth_counts[i]

    position_ids = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]

    # start = 0
    # for i in range(len(depth_counts)):
    #     position_ids[start: start + depth_counts[i]] = i
    #     start += depth_counts[i]

    tree_buffers = {
        "attn_mask": [i.unsqueeze(0).unsqueeze(0) for i in tree_attn_mask_list],
        "tree_indices": tree_indices_list,
        "position_ids":position_ids,
        "repeat_nums":repeat_nums
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: [i.clone().to(device) for i in v]
        if isinstance(v[0], torch.Tensor)
        else (
            torch.tensor(v, device=device)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in tree_buffers.items()
    }
    return tree_buffers


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def _to_numpy_flat(x, force_float=False):
    """Convert list or tensor to flat numpy array. Returns (arr, shape, device, dtype).
    force_float: if True, convert to float64 (for confidence); if False, preserve dtype (for labels).
    """
    if isinstance(x, torch.Tensor):
        shape = x.shape
        device = x.device
        dtype = x.dtype
        arr = x.detach().cpu().numpy().flatten()
        if force_float:
            arr = arr.astype(np.float64)
        return arr, shape, device, dtype
    else:
        arr = np.array(x).flatten()
        if force_float:
            arr = arr.astype(np.float64)
        return arr, arr.shape, None, None


def _from_numpy_to_tensor(arr, shape, device=None, dtype=None):
    """Convert numpy array to tensor, reshape to original shape."""
    out = torch.from_numpy(arr.astype(np.float32))
    if shape is not None:
        out = out.reshape(shape)
    if device is not None:
        out = out.to(device)
    if dtype is not None:
        out = out.to(dtype)
    return out


class Calibration:
    def __init__(self, method="platt", n_bins=15):
        """
        method: "histogram" or "platt"
        n_bins: number of bins for histogram binning
        Accepts list or tensor (any shape) for fit/transform. transform always returns tensor.
        """
        self.method = method
        self.n_bins = n_bins
        self.bin_edges = None
        self.bin_acc = None
        self.platt_model = LogisticRegression()

    # -------------------------------------------------
    # Fit
    # -------------------------------------------------
    def fit(self, pred_label, gt_label, pred_confidence):
        pred_label, _, _, _ = _to_numpy_flat(pred_label, force_float=False)
        gt_label, _, _, _ = _to_numpy_flat(gt_label, force_float=False)
        pred_confidence, _, _, _ = _to_numpy_flat(pred_confidence, force_float=True)

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

        self.platt_model.fit(logits, correct)

    def _transform_platt(self, confidences):
        eps = 1e-6
        confidences = np.clip(confidences, eps, 1 - eps)
        logits = np.log(confidences / (1 - confidences)).reshape(-1, 1)

        calibrated = self.platt_model.predict_proba(logits)[:, 1]
        return calibrated

    # -------------------------------------------------
    # Public transform: accepts list or tensor (any shape), returns tensor
    # -------------------------------------------------
    def transform(self, pred_confidence):
        arr, shape, device, dtype = _to_numpy_flat(pred_confidence, force_float=True)

        if self.method == "histogram":
            calibrated = self._transform_histogram(arr)
        elif self.method == "platt":
            calibrated = self._transform_platt(arr)
        else:
            raise ValueError("method must be 'histogram' or 'platt'")

        return _from_numpy_to_tensor(calibrated, shape, device, dtype)

    # -------------------------------------------------
    # ECE
    # -------------------------------------------------
    def compute_ece(self, pred_label, gt_label, pred_confidence):
        pred_label, _, _, _ = _to_numpy_flat(pred_label, force_float=False)
        gt_label, _, _, _ = _to_numpy_flat(gt_label, force_float=False)
        pred_confidence, _, _, _ = _to_numpy_flat(pred_confidence, force_float=True)

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




if __name__=="__main__":
    from choices import mc_sim_7b_63
    a=generate_tree_buffers(mc_sim_7b_63)
    print(a)