#Cell 1
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from scipy.ndimage import label, find_objects, maximum_filter, gaussian_filter
from sklearn.metrics import precision_recall_curve, auc
from yy_maker_custom_level_scheme import LoadMat

#Cell 2
# Patch size, since full matrix is too large to process at once
PATCH_SIZE = 64
# How much more to penalize missing a peak
POS_WEIGHT = 50.0

#Cell 3 -- Attention UNet Code
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from scipy.ndimage import label, find_objects, maximum_filter, gaussian_filter
from yy_maker_custom_level_scheme import LoadMat


PATCH_SIZE = 64
POS_WEIGHT = 50.0
LABEL_SIGMA = 0.5
OFFSET_RADIUS = 6
LAMBDA_OFFSET = 2.0

class CoincidenceDataset(IterableDataset):
    """
    Loads one matrix pair from disk at a time, cuts `patches_per_mat` patches from it, then lets
    it be garbage collected, moving to the next.
    Only one matrix is ever resident in memory, instead of all of them at once.
    """
    def __init__(self, data_dir=None, patches_per_mat=100, patch_size=PATCH_SIZE,
                 n_matrices=None, file_pairs=None):
        self.P   = patch_size
        self.ppm = patches_per_mat

        if file_pairs is not None:
            self.file_pairs = file_pairs
        else:
            noisy_dir = os.path.join(data_dir, "NonBlurredArrays")
            files = sorted([
                f for f in os.listdir(noisy_dir)
                if f.startswith("yy_coincidence_matrix(") and "clean" not in f
            ])
            if n_matrices is not None:
                files = files[:n_matrices]
            if not files:
                raise FileNotFoundError(f"No matrix files found in {noisy_dir}")

            self.file_pairs = []
            for fname in files:
                idx    = fname.replace("yy_coincidence_matrix(", "").replace(").mat", "")
                s_path = os.path.join(noisy_dir, fname)
                c_path = os.path.join(noisy_dir, f"yy_coincidence_matrix_clean({idx}).mat")
                if not os.path.exists(c_path):
                    print(f"  skipping {fname} -- clean file missing")
                    continue
                self.file_pairs.append((s_path, c_path))

        print(f"Dataset ready: {len(self.file_pairs)} matrices x {self.ppm} patches "
              f"= {len(self)} total patches per epoch\n")

    def __len__(self):
        return len(self.file_pairs) * self.ppm

    def _patches_from_pair(self, s_path, c_path):
        '''scattered = LoadMat(s_path)
        clean     = LoadMat(c_path)
        norm      = _normalise(scattered)
        heatmap, offset, mask = _make_labels(clean)
        peaks     = _get_peak_positions(clean)
        size      = norm.shape[0]'''
        scattered = LoadMat(s_path)
        clean     = LoadMat(c_path)
        pad       = self.P // 2
        norm      = np.pad(_normalise(scattered), pad, mode='reflect')
        clean     = np.pad(clean, pad, mode='reflect')
        heatmap, offset, mask = _make_labels(clean)
        peaks     = _get_peak_positions(clean)
        size      = norm.shape[0]

        '''for _ in range(self.ppm):
            if np.random.random() < 0.5 and peaks:
                pr, pc = peaks[np.random.randint(len(peaks))]
                jitter = self.P // 4
                r = int(np.clip(pr - self.P // 2 + np.random.randint(-jitter, jitter + 1),
                                0, size - self.P))
                c = int(np.clip(pc - self.P // 2 + np.random.randint(-jitter, jitter + 1),
                                0, size - self.P))
            else:
                r = np.random.randint(0, size - self.P)
                c = np.random.randint(0, size - self.P)'''
        
        for patch_idx in range(self.ppm):
            if patch_idx < 2:
                # Force an edge patch covering the axis region
                if np.random.random() < 0.5:
                    r = 0  # top edge
                    c = np.random.randint(0, size - self.P)
                else:
                    r = np.random.randint(0, size - self.P)
                    c = 0  # left edge
            elif np.random.random() < 0.5 and peaks:
                pr, pc = peaks[np.random.randint(len(peaks))]
                jitter = self.P // 4
                r = int(np.clip(pr - self.P // 2 + np.random.randint(-jitter, jitter + 1),
                                0, size - self.P))
                c = int(np.clip(pc - self.P // 2 + np.random.randint(-jitter, jitter + 1),
                                0, size - self.P))
            else:
                r = np.random.randint(0, size - self.P)
                c = np.random.randint(0, size - self.P)

            x = torch.from_numpy(norm   [r:r+self.P, c:c+self.P]).unsqueeze(0)
            y_heat = torch.from_numpy(heatmap[r:r+self.P, c:c+self.P]).unsqueeze(0)
            y_off = torch.from_numpy(offset [:, r:r+self.P, c:c+self.P])
            y_mask = torch.from_numpy(mask   [:, r:r+self.P, c:c+self.P])
            yield x, y_heat, y_off, y_mask
        # scattered/clean/norm/heatmap/offset/mask/peaks fall out of scope here

    def __iter__(self):
        order = np.random.permutation(len(self.file_pairs))
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:            # split matrices across workers if you use num_workers>0
            order = order[worker_info.id::worker_info.num_workers]
        for i in order:
            s_path, c_path = self.file_pairs[i]
            yield from self._patches_from_pair(s_path, c_path)


class _DoubleConv(nn.Module):
    """Two consecutive 3x3 conv layers, each followed by BatchNorm + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class BottleneckAttention(nn.Module):
    """
    Multi-head self-attention applied after the bottleneck DoubleConv.

    At the bottleneck the spatial size is 4x4 for a 64x64 input (only 16
    positions), so full self-attention is cheap. It lets every bottleneck
    cell attend to every other, allowing the compressed representation to
    maintain distinct signals for two nearby peaks that the encoder partially
    mixed — the model can learn that two nearby activations should stay
    separate rather than collapse into one blob.

    A residual connection preserves the convolutional features so the
    attention only needs to correct, not reconstruct.
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).permute(0, 2, 1)          # (B, H*W, C)
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.norm(seq + attn_out)               # residual + norm
        return seq.permute(0, 2, 1).contiguous().reshape(B, C, H, W)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,  F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,  F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        alpha = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * alpha

class AttentionUNet(nn.Module):
    """
    U-Net with three improvements over the plain baseline:

    1. Attention gates on every skip connection (decoder).
       Suppress background activations in encoder features before they reach
       the decoder — reduces false positives.

    2. Self-attention at the bottleneck (BottleneckAttention).
       Lets the 4x4 compressed representation maintain distinct signals for
       nearby peaks the encoder may have partially merged — helps with peak
       separation.

    3. Dual output heads: heatmap + offset.
       - heatmap head: 1 channel, trained with BCE.  "Is there a peak here?"
       - offset head:  2 channels (dr, dc), trained with L1.  "How far to the
         nearest true peak center?"  Used at inference to refine localisation
         to sub-pixel precision, eliminating the 1-2 px offset seen with plain
         max filter detection.

    Returns: (heatmap_logits, offset_map)  both shaped [B, *, H, W].
    """
    def __init__(self, features=(32, 64, 128, 256)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.downs = nn.ModuleList()
        in_ch = 1
        for f in features:
            self.downs.append(_DoubleConv(in_ch, f))
            in_ch = f

        self.bottleneck      = _DoubleConv(features[-1], features[-1] * 2)
        self.bottleneck_attn = BottleneckAttention(features[-1] * 2, num_heads=8)

        self.ups   = nn.ModuleList()
        self.gates = nn.ModuleList()
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.gates.append(AttentionGate(F_g=f, F_l=f, F_int=f // 2))
            self.ups.append(_DoubleConv(f * 2, f))

        self.heatmap_head = nn.Conv2d(features[0], 1, kernel_size=1)
        self.offset_head  = nn.Conv2d(features[0], 2, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.bottleneck_attn(x)

        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[i // 2]
            if x.shape != s.shape:
                x = F.interpolate(x, size=s.shape[2:])
            s = self.gates[i // 2](g=x, x=s)
            x = torch.cat([s, x], dim=1)
            x = self.ups[i + 1](x)

        return self.heatmap_head(x), self.offset_head(x)


def _normalise(mat):
    out = np.log1p(mat)
    mx  = out.max()
    return (out / mx).astype(np.float32) if mx > 0 else out.astype(np.float32)


def _get_peak_positions(clean_mat):
    """Intensity-weighted centroid of each blob in the unblurred clean matrix."""
    '''binary, _ = label(clean_mat > 0)
    objects   = find_objects(binary)
    peaks = []
    for obj in objects:
        r_sl, c_sl = obj
        region = clean_mat[r_sl, c_sl] * (binary[r_sl, c_sl] > 0)
        rr = np.arange(r_sl.start, r_sl.stop)
        cc = np.arange(c_sl.start, c_sl.stop)
        RR, CC = np.meshgrid(rr, cc, indexing='ij')
        w = region.sum()
        if w > 0:
            peaks.append((int(round((RR * region).sum() / w)),
                          int(round((CC * region).sum() / w))))'''
    rows, cols = np.where(clean_mat > 0)
    true_peaks = list(zip(rows.tolist(), cols.tolist()))
    return true_peaks


def _make_labels(clean_mat, sigma=LABEL_SIGMA, offset_radius=OFFSET_RADIUS):
    """
    Generate heatmap and offset labels from the *unblurred* clean matrix.

    Heatmap: narrow Gaussian (sigma=0.5) centred at each true peak. Narrow
             enough that peaks 3+ px apart have < 0.01% label overlap —
             the model is never trained on merged targets.

    Offset:  for each pixel within offset_radius of any true peak, the label
             is (peak_row - pixel_row, peak_col - pixel_col). Background pixels
             receive (0, 0) and are excluded from the loss via the mask.

    Mask:    1 within offset_radius of any true peak, 0 elsewhere.
    """
    H, W  = clean_mat.shape
    peaks = _get_peak_positions(clean_mat)

    # Heatmap: point image blurred with narrow Gaussian
    point_img = np.zeros((H, W), dtype=np.float32)
    for pr, pc in peaks:
        point_img[pr, pc] = 1.0
    heatmap = gaussian_filter(point_img, sigma=sigma)
    mx = heatmap.max()
    heatmap = (heatmap / mx).astype(np.float32) if mx > 0 else heatmap

    # Offset labels and mask
    offset = np.zeros((2, H, W), dtype=np.float32)
    mask   = np.zeros((1, H, W), dtype=np.float32)
    rows_g, cols_g = np.mgrid[0:H, 0:W]

    for pr, pc in peaks:
        r0, r1 = max(0, pr - offset_radius), min(H, pr + offset_radius + 1)
        c0, c1 = max(0, pc - offset_radius), min(W, pc + offset_radius + 1)
        dr   = pr - rows_g[r0:r1, c0:c1]
        dc   = pc - cols_g[r0:r1, c0:c1]
        dist = np.sqrt(dr**2 + dc**2)
        in_r = dist <= offset_radius

        # Only overwrite if this peak is closer than whichever peak was written first
        existing = np.sqrt(offset[0, r0:r1, c0:c1]**2 + offset[1, r0:r1, c0:c1]**2)
        closer   = in_r & ((mask[0, r0:r1, c0:c1] == 0) | (dist < existing))

        offset[0, r0:r1, c0:c1] = np.where(closer, dr, offset[0, r0:r1, c0:c1])
        offset[1, r0:r1, c0:c1] = np.where(closer, dc, offset[1, r0:r1, c0:c1])
        mask  [0, r0:r1, c0:c1] = np.where(closer, 1.0, mask[0, r0:r1, c0:c1])

    return heatmap, offset, mask


def combined_loss(heatmap_pred, offset_pred, y_heat, y_off, y_mask,
                  pos_weight=POS_WEIGHT, lambda_offset=LAMBDA_OFFSET):
    """
    BCE loss on heatmap + L1 loss on offset (at peak pixels only).

    Why L1 for the offset?  BCE near a true peak plateau gives weak gradient
    for small spatial offsets — the model can predict "there is a peak here"
    correctly while still being 1-2 px off.  L1 penalises that offset linearly
    and keeps the gradient non-zero all the way to zero error, pushing the
    prediction toward the exact center.

    The L1 term is computed only where y_mask == 1 (within OFFSET_RADIUS of
    a true peak) so background pixels (where offset is undefined) don't
    dominate or add noisy gradients.
    """
    pw  = torch.tensor([pos_weight], device=heatmap_pred.device)
    bce = F.binary_cross_entropy_with_logits(heatmap_pred, y_heat, pos_weight=pw)

    n_mask = y_mask.sum().clamp(min=1)
    l1     = (F.l1_loss(offset_pred * y_mask, y_off * y_mask, reduction='sum') / n_mask)

    return bce + lambda_offset * l1


def _train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for x, y_heat, y_off, y_mask in loader:
        x, y_heat, y_off, y_mask = (t.to(device) for t in (x, y_heat, y_off, y_mask))
        heat_pred, off_pred = model(x)
        loss = combined_loss(heat_pred, off_pred, y_heat, y_off, y_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def _test_epoch(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y_heat, y_off, y_mask in loader:
            x, y_heat, y_off, y_mask = (t.to(device) for t in (x, y_heat, y_off, y_mask))
            heat_pred, off_pred = model(x)
            total += combined_loss(heat_pred, off_pred, y_heat, y_off, y_mask).item()
    return total / len(loader)


def run_training(data_dir="./data", n_epochs=20, batch_size=32, lr=1e-3,
                 patches_per_mat=100, n_matrices=None,
                 save_path="peak_detector_attn.pt"):

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps'  if torch.backends.mps.is_available()
                          else 'cpu')
    print(f"Device: {device}\n")

    full    = CoincidenceDataset(data_dir, patches_per_mat, n_matrices=n_matrices)
    n_train = int(0.8 * len(full.file_pairs))
    train_pairs, test_pairs = full.file_pairs[:n_train], full.file_pairs[n_train:]
    print(f"Split: {len(train_pairs)} train / {len(test_pairs)} test matrices")

    train_ds = CoincidenceDataset(file_pairs=train_pairs, patches_per_mat=patches_per_mat)
    test_ds  = CoincidenceDataset(file_pairs=test_pairs,  patches_per_mat=patches_per_mat)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, num_workers=0)

    model     = AttentionUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    print(f"\nTraining AttentionUNet for {n_epochs} epochs (BCE + L1 loss):")
    train_losses, test_losses = [], []
    for epoch in range(1, n_epochs + 1):
        tr = _train_epoch(model, train_loader, optimizer, device)
        te = _test_epoch(model, test_loader,  device)
        scheduler.step()
        train_losses.append(tr)
        test_losses.append(te)
        print(f"  Epoch {epoch:3d}/{n_epochs}  train={tr:.5f}  test={te:.5f}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, n_epochs + 1), train_losses, label="Train")
    plt.plot(range(1, n_epochs + 1), test_losses,  label="Test")
    plt.xlabel("Epoch"); plt.ylabel("BCE + L1 Loss")
    plt.title("Attention U-Net -- Training Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("training_loss_attn.png", dpi=150)

    model.cpu()
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved -> {save_path}")
    return model



def _build_heatmap_and_offset(model, mat, patch_size=PATCH_SIZE, stride=16):
    """Returns (heatmap, offset_map) both at full matrix resolution."""
    device  = next(model.parameters()).device
    model.eval()

    size    = mat.shape[0]
    mat_n   = _normalise(mat)

    pad      = patch_size // 2
    padded   = np.pad(mat_n, pad, mode='reflect')
    pad_size = padded.shape[0]

    heat   = np.zeros((pad_size, pad_size),    dtype=np.float32)
    offset = np.zeros((2, pad_size, pad_size), dtype=np.float32)
    count  = np.zeros((pad_size, pad_size),    dtype=np.float32)

    rows = list(range(0, pad_size - patch_size, stride)) + [pad_size - patch_size]
    cols = list(range(0, pad_size - patch_size, stride)) + [pad_size - patch_size]

    with torch.no_grad():
        for r in rows:
            for c in cols:
                patch = padded[r:r+patch_size, c:c+patch_size]
                x     = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                h_out, o_out = model(x)
                heat  [r:r+patch_size, c:c+patch_size]    += torch.sigmoid(h_out).squeeze().cpu().numpy()
                offset[:, r:r+patch_size, c:c+patch_size] += o_out.squeeze().cpu().numpy()
                count [r:r+patch_size, c:c+patch_size]    += 1

    cnt = np.maximum(count, 1)
    heatmap    = heat  [   pad:pad+size, pad:pad+size] / cnt[pad:pad+size, pad:pad+size]
    offset_map = offset[:, pad:pad+size, pad:pad+size] / cnt[pad:pad+size, pad:pad+size]
    return heatmap, offset_map


def _build_heatmap(model, mat, patch_size=PATCH_SIZE, stride=16):
    """Backward-compatible wrapper — returns only the heatmap."""
    heatmap, _ = _build_heatmap_and_offset(model, mat, patch_size, stride)
    return heatmap


def detect_peaks(model, mat, patch_size=PATCH_SIZE, stride=32,
                 threshold=0.5, min_distance=5, min_peak_pixels=4):
    """
    Blob-gated local-maximum detection with offset refinement.

    Pipeline:
      1. Threshold heatmap -> connected blobs (same as centroid/maxfilter methods)
      2. Discard blobs smaller than min_peak_pixels (noise rejection)
      3. Within each valid blob, find local maxima spaced min_distance apart
      4. Shift each local maximum by the predicted (dr, dc) offset to get the
         refined coordinate — eliminates the 1-2 px localisation error

    Gating through blobs is critical: the offset head was only trained at pixels
    near true peaks. At background noise pixels the offset output is essentially
    random, so applying it globally (without blob segmentation) causes detections
    to scatter to arbitrary locations.
    """
    heatmap, offset_map = _build_heatmap_and_offset(model, mat, patch_size, stride)
    size = mat.shape[0]

    binary, _ = label(heatmap > threshold)
    objects   = find_objects(binary)

    peaks = []
    for obj in objects:
        r_sl, c_sl = obj
        blob_mask  = (binary[r_sl, c_sl] > 0).astype(np.float32)
        if blob_mask.sum() < min_peak_pixels:
            continue

        region    = heatmap[r_sl, c_sl] * blob_mask
        local_max = (maximum_filter(region, size=min_distance) == region) & (blob_mask > 0)

        for rm, cm in np.argwhere(local_max):
            r = r_sl.start + rm
            c = c_sl.start + cm
            dr = float(offset_map[0, r, c])
            dc = float(offset_map[1, r, c])
            rr = int(round(r + dr))
            cc = int(round(c + dc))
            rr = max(0, min(size - 1, rr))
            cc = max(0, min(size - 1, cc))
            peaks.append((rr, cc))

    print(f"Detected {len(peaks)} peaks (threshold={threshold})")
    return peaks, heatmap



def classify_peaks_close(detected_peaks, true_peaks, tolerance=0):
    """
    Greedy one-to-one matching between detected and true peaks.
    Identical to the fixed version in cnn.ipynb.
    """
    true_arr     = np.array(true_peaks) if true_peaks else np.empty((0, 2))
    matched_true = set()
    true_positives, closer_positives, false_positives = [], [], []
    det_to_true_idx = {}

    for i, det in enumerate(detected_peaks):
        if len(true_arr) == 0:
            false_positives.append(det)
            continue
        dists = np.sqrt(((true_arr - np.array(det)) ** 2).sum(axis=1))
        best_idx = best_dist = None
        for idx in np.argsort(dists):
            if dists[idx] > tolerance:
                break
            if idx not in matched_true:
                best_idx, best_dist = int(idx), dists[idx]
                break
        if best_idx is not None and best_dist == 0:
            true_positives.append(det)
            matched_true.add(best_idx)
            det_to_true_idx[i] = best_idx
        elif best_idx is not None:
            closer_positives.append(det)
            matched_true.add(best_idx)
            det_to_true_idx[i] = best_idx
        else:
            false_positives.append(det)

    false_negatives = [true_peaks[i] for i in range(len(true_peaks))
                       if i not in matched_true]
    return true_positives, false_positives, false_negatives, closer_positives, det_to_true_idx


def evaluate_detection(detected_peaks, true_peaks, tolerances=(0,)):
    true_arr = np.array(true_peaks)
    det_arr  = np.array(detected_peaks) if detected_peaks else np.empty((0, 2))
    print(f"{'Tol':>5} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 55)
    for tol in tolerances:
        matched = set()
        tp = fp = 0
        for det in det_arr:
            if len(true_arr) == 0:
                fp += 1; continue
            dists = np.sqrt(((true_arr - det) ** 2).sum(axis=1))
            best  = int(dists.argmin())
            if dists[best] <= tol and best not in matched:
                tp += 1; matched.add(best)
            else:
                fp += 1
        fn   = len(true_peaks) - len(matched)
        prec = tp / (tp + fp)        if tp + fp > 0 else 0.0
        rec  = tp / (tp + fn)        if tp + fn > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if prec + rec > 0 else 0.0
        print(f"{tol:>5} {tp:>6} {fp:>6} {fn:>6} {prec:>10.3f} {rec:>8.3f} {f1:>8.3f}")


def plot_results(mat, heatmap, true_pos, false_pos, false_neg,
                 save_path="detected_peaks_attn.png"):
    _, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(np.log1p(mat), cmap='hot', origin='lower', aspect='auto')
    axes[0].set_title("Coincidence matrix")
    axes[1].imshow(heatmap, cmap='inferno', origin='lower', aspect='auto',
                   norm=mcolors.PowerNorm(gamma=0.5))
    for r, c in true_pos:
        axes[1].plot(c, r, 'x', color='green', ms=8, mew=1.5, label='TP')
    for r, c in false_pos:
        axes[1].plot(c, r, 'x', color='red',   ms=8, mew=1.5, label='FP')
    for r, c in false_neg:
        axes[1].plot(c, r, 'o', color='blue',  ms=6, mew=1.5, fillstyle='none', label='FN')
    handles, labels_list = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels_list, handles))
    axes[1].legend(by_label.values(), by_label.keys(), loc='upper right')
    n_det = len(true_pos) + len(false_pos)
    axes[1].set_title(f"Heatmap -- {n_det} detected, {len(false_neg)} missed")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    model1 = run_training(
        data_dir = "../../data",
        n_epochs = 400,
        batch_size = 32,
        lr = 1e-3,
        patches_per_mat = 25,
        n_matrices = 300,
        save_path = "peak_detector_attn_500.pt",
    )

    '''with open("yy_coincidence_matrix.mat", "rb") as f:
        mat = pickle.load(f)
    with open("yy_coincidence_matrix_clean.mat", "rb") as f:
        clean_mat = pickle.load(f)
    
    rows, cols = np.where(clean_mat > 0)
    true_peaks = list(zip(rows.tolist(), cols.tolist()))

    print("\n--- Offset-refined detection (new) ---")
    peaks_o, heatmap = detect_peaks(model1, mat, threshold=0.7, min_distance=4)
    evaluate_detection(peaks_o, true_peaks)
    tp, fp, fn, closer, _ = classify_peaks_close(peaks_o, true_peaks, tolerance=0)
    plot_results(mat, heatmap, tp, fp, fn, save_path="results_offset_attn.png")'''

#Cell 4 -- Testing on 50 matrices
import os
import json

test_dir = "test_matrices/NonBlurredArrays"

# Find all scattered matrix files in the test folder
test_files = sorted([
    f for f in os.listdir(test_dir)
    if f.startswith("yy_coincidence_matrix(") and "clean" not in f
])

all_results = {}

for fname in test_files:
    idx = fname.replace("yy_coincidence_matrix(", "").replace(").mat", "")
    s_path = os.path.join(test_dir, fname)
    c_path = os.path.join(test_dir, f"yy_coincidence_matrix_clean({idx}).mat")

    if not os.path.exists(c_path):
        print(f"Skipping {fname} — clean file missing")
        continue

    print(f"\n=== Matrix {idx} ===")

    with open(s_path, "rb") as f:
        mat = pickle.load(f)
    with open(c_path, "rb") as f:
        clean_mat = pickle.load(f)

    rows, cols = np.where(clean_mat > 0)
    true_peaks = list(zip(rows.tolist(), cols.tolist()))

    peaks_o, heatmap = detect_peaks(model1, mat, threshold=0.7, min_distance=4)
    evaluate_detection(peaks_o, true_peaks)

    tp, fp, fn, closer, _ = classify_peaks_close(peaks_o, true_peaks, tolerance=0)
    plot_results(mat, heatmap, tp, fp, fn,
                 save_path=f"results_matrix_{idx}.png")

    all_results[idx] = {
        "true_peaks": true_peaks,
        "detected_peaks": peaks_o,
        "n_true": len(true_peaks),
        "n_detected": len(peaks_o),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
    }

# Save all peaks and metrics to a single JSON file
with open("detection_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved all peaks and metrics -> detection_results.json")

#Cell 5 -- Print True vs Detected Peaks
true_peaks = [tuple(p) for p in all_results["19"]["true_peaks"]]
detected_peaks = [tuple(p) for p in all_results["19"]["detected_peaks"]]

import sys

print(f"{'TRUE PEAKS':<20} {'DETECTED PEAKS':<20}")
print("-" * 40)
for i in range(max(len(true_peaks), len(peaks_o))):
    tp = f"({true_peaks[i][0]}, {true_peaks[i][1]})" if i < len(true_peaks) else ""
    '''dp = f"({peaks_o[i][0]}, {peaks_o[i][1]})"           if i < len(peaks_o)      else ""'''
    dp = f"({detected_peaks[i][0]}, {detected_peaks[i][1]})"           if i < len(detected_peaks)      else ""
    print(f"{tp:<20} {dp:<20}", file=sys.stderr)

#Cell 6 -- Find Matrices with peaks 2 pixels or closer
import json
import numpy as np
from scipy.spatial import cKDTree

with open("detection_results_v2.json") as f:
    all_results = json.load(f)

for idx in sorted(all_results.keys(), key=lambda x: int(x)):
    peaks = [tuple(p) for p in all_results[idx]["detected_peaks"]]
    if not peaks:
        continue

    arr = np.array(peaks)
    tree = cKDTree(arr)
    pairs = tree.query_pairs(r=2.0)

    if pairs:
        print(f"\n=== Matrix {idx} — {len(pairs)} pair(s) within 2 pixels ===")
        for i, j in sorted(pairs):
            dist = round(float(np.linalg.norm(arr[i] - arr[j])), 3)
            print(f"  {tuple(arr[i])}  <->  {tuple(arr[j])}  (dist={dist})")
    else:
        print(f"Matrix {idx}: no close pairs")

#Cell 7 -- Print FNs
detected = set(detected_peaks)
true = set(true_peaks)

only_in_true = sorted(true - detected)
only_in_either = sorted(detected ^ true)

print("\nFN:")
for p in only_in_true:
    print(p)

#Cell 8 -- Looking closely at heatmap
def plot_offset_crosssection(heatmap, offset_map, peak_a, peak_b,
                             margin=15, save_path="crosssection_offset.png"):
    r0 = max(0, min(peak_a[0], peak_b[0]) - margin)
    r1 = min(heatmap.shape[0], max(peak_a[0], peak_b[0]) + margin)
    c0 = max(0, min(peak_a[1], peak_b[1]) - margin)
    c1 = min(heatmap.shape[1], max(peak_a[1], peak_b[1]) + margin)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    zoom_heat = heatmap[r0:r1, c0:c1]
    dr_z = offset_map[0, r0:r1, c0:c1]
    dc_z = offset_map[1, r0:r1, c0:c1]

    rows_z, cols_z = np.mgrid[0:r1-r0, 0:c1-c0]
    active = zoom_heat > 0.2

    axes[0].imshow(zoom_heat, cmap='inferno', origin='lower')
    axes[0].quiver(cols_z[active], rows_z[active],
                   dc_z[active], dr_z[active],
                   color='white', alpha=0.7, scale=25, width=0.004)
    axes[0].plot(peak_a[1]-c0, peak_a[0]-r0, 'b+', ms=14, mew=2.5, label=f'True A {peak_a}')
    axes[0].plot(peak_b[1]-c0, peak_b[0]-r0, 'g+', ms=14, mew=2.5, label=f'True B {peak_b}')
    axes[0].set_title("Heatmap + predicted offset field")
    axes[0].legend(fontsize=8)

    shared_row = (peak_a[0] + peak_b[0]) // 2
    col_range  = np.arange(c0, c1)
    probs      = heatmap[shared_row, c0:c1]
    dc_slice   = offset_map[1, shared_row, c0:c1]

    ax1 = axes[1]
    ax2 = ax1.twinx()

    ax1.plot(col_range, probs, color='orange', linewidth=2, label='CNN probability')
    ax1.axvline(peak_a[1], color='blue',  linestyle='--', lw=1.5, label=f'True A ({peak_a[1]})')
    ax1.axvline(peak_b[1], color='green', linestyle='--', lw=1.5, label=f'True B ({peak_b[1]})')
    ax1.set_ylabel('CNN probability', color='orange')
    ax1.set_xlabel('Column (pixels)')

    ax2.plot(col_range, dc_slice, color='cyan', linewidth=2, linestyle=':', label='Col offset (dc)')
    ax2.axhline(0, color='cyan', linewidth=0.5, linestyle='--')
    ax2.set_ylabel('Predicted column offset dc (px)', color='cyan')

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc='upper right', fontsize=8)
    ax1.set_title(f"1D slice at row {shared_row}  —  dc flips at peak center")

    plt.suptitle("Attention U-Net: heatmap + offset field", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_offset_zoom(heatmap, offset_map, peak_a, peak_b,
                     threshold=0.5, min_distance=5, min_peak_pixels=4,
                     margin=20, save_path='offset_zoom.png'):
    r0 = max(0, min(peak_a[0], peak_b[0]) - margin)
    r1 = min(heatmap.shape[0], max(peak_a[0], peak_b[0]) + margin)
    c0 = max(0, min(peak_a[1], peak_b[1]) - margin)
    c1 = min(heatmap.shape[1], max(peak_a[1], peak_b[1]) + margin)

    zoom = heatmap[r0:r1, c0:c1]

    binary, _ = label(zoom > threshold)
    objects   = find_objects(binary)

    raw_maxima, refined = [], []
    for obj in objects:
        rs, cs = obj
        blob_mask = (binary[rs, cs] > 0).astype(np.float32)
        blob = zoom[rs, cs] * blob_mask
        if blob_mask.sum() < min_peak_pixels:
            continue
        lmax = (maximum_filter(blob, size=min_distance) == blob) & (blob_mask > 0)
        for rm, cm in np.argwhere(lmax):
            r_g = rm + rs.start + r0
            c_g = cm + cs.start + c0
            raw_maxima.append((r_g, c_g))
            dr = float(offset_map[0, r_g, c_g])
            dc = float(offset_map[1, r_g, c_g])
            refined.append((int(round(r_g + dr)), int(round(c_g + dc))))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.imshow(zoom, cmap='inferno', origin='lower', aspect='auto',
              extent=[c0, c1, r0, r1])

    ax.plot(peak_a[1], peak_a[0], 'b+', ms=16, mew=2.5, label=f'True A {peak_a}')
    ax.plot(peak_b[1], peak_b[0], 'g+', ms=16, mew=2.5, label=f'True B {peak_b}')

    for i, (r, c) in enumerate(raw_maxima):
        ax.plot(c, r, 'wo', ms=10, mew=1.5, fillstyle='none',
                label='Raw local max' if i == 0 else '')

    for i, (r, c) in enumerate(refined):
        ax.plot(c, r, 'rx', ms=12, mew=2.5,
                label='Detected' if i == 0 else '')

    for (r_raw, c_raw), (r_ref, c_ref) in zip(raw_maxima, refined):
        ax.annotate('', xy=(c_ref, r_ref), xytext=(c_raw, r_raw),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2.0))

    ax.set_title("Heatmap (zoomed)  —  ○ raw max  →  × detection")
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1]
    shared_row = (peak_a[0] + peak_b[0]) // 2
    col_range = np.arange(c0, c1)
    probs = heatmap[shared_row, c0:c1]

    ax.plot(col_range, probs, color='orange', linewidth=2, label='CNN probability')
    ax.axhline(threshold, color='grey', linestyle='--', lw=1, label=f'Threshold ({threshold})')
    ax.axvline(peak_a[1], color='blue',  linestyle='--', lw=1.5, label=f'True A ({peak_a[1]})')
    ax.axvline(peak_b[1], color='green', linestyle='--', lw=1.5, label=f'True B ({peak_b[1]})')

    for i, (r, c) in enumerate(raw_maxima):
        ax.axvline(c, color='white', linestyle=':', lw=1.5,
                   label=f'Raw max ({c})' if i == 0 else f'Raw max ({c})')

    for i, (r, c) in enumerate(refined):
        ax.axvline(c, color='red', linestyle='-', lw=1.5,
                   label=f'Refined ({c})' if i == 0 else f'Refined ({c})')

    ax.set_title(f"1D cross-section at row {shared_row}")
    ax.set_xlabel("Column (pixels)")
    ax.set_ylabel("CNN probability")
    ax.legend(loc='upper right', fontsize=8)

    sep = abs(peak_b[1] - peak_a[1])
    plt.suptitle(f"Offset-refined detection  —  peak separation: {sep} px",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Raw local maxima : {raw_maxima}")
    print(f"Offset-refined   : {refined}")
    print(f"True peaks       : {peak_a}, {peak_b}")


heatmap, offset_map = _build_heatmap_and_offset(model1, mat)

plot_offset_crosssection(heatmap, offset_map, peak_a=(935, 1555), peak_b=(935, 1556))
plot_offset_zoom(heatmap, offset_map, peak_a=(935, 1555), peak_b=(935, 1556))

plot_offset_zoom(heatmap, offset_map, peak_a=(935, 1555), peak_b=(935, 1556))
