import os
import matplotlib.pyplot as plt

def plot_branching_level_scheme(transition_scheme, fig_name="Level Scheme", out_dir="./data/LevelSchemes"):
    n = len(transition_scheme)
    
    abs_energy = [0] * n
    for i in range(1, n):
        g_e      = int(transition_scheme[i][0])
        next_idx = int(transition_scheme[i][1])
        if next_idx >= 0:
            abs_energy[i] = abs_energy[next_idx] + g_e
    energy_max = max(abs_energy) if max(abs_energy) > 0 else 1

    children = {i: [] for i in range(n)}
    for ev, next_idx, idx in transition_scheme:
        next_idx = int(next_idx)
        idx = int(idx)
        if next_idx != -1:
            children[next_idx].append(idx)

    def subtree_width(node):
        if not children[node]:
            return 1
        return sum(subtree_width(c) for c in children[node])

    x_pos = {}
    def assign_x(node, x_start, x_end):
        x_pos[node] = (x_start + x_end) / 2
        if not children[node]:
            return
        total_w = sum(subtree_width(c) for c in children[node])
        cx = x_start
        for child in children[node]:
            w = subtree_width(child) / total_w * (x_end - x_start)
            assign_x(child, cx, cx + w)
            cx += w

    assign_x(0, 0.0, 1.0)

    y_pad = 0.04
    def to_y(idx):
        return y_pad + idx / max(n - 1, 1) * (1 - 2 * y_pad)

    fig_h = max(10, n * 0.35)
    fig_w = max(10, subtree_width(0) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    lbl_box = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9)
    inv_lbl_box = dict(boxstyle='round,pad=0.15', fc='#ffeeee', ec='none', alpha=0.9)
    colors = ['#8e44ad','#2980b9','#27ae60','#e67e22','#c0392b',
              '#1abc9c','#d35400','#34495e','#f39c12','#16a085']
    hw = 0.018

    # Draw level lines and energy labels
    for ev, next_idx, idx in transition_scheme:
        idx = int(idx)
        ev = int(ev)
        x = x_pos[idx]
        y = to_y(idx)
        lw = 2.5 if idx == 0 else 1.5
        ax.hlines(y, x - hw, x + hw, colors='black', linewidth=lw, zorder=3)
        label = "GS  0 keV" if idx == 0 else f"{abs_energy[idx]} keV  (idx {idx})"
        ax.text(x + hw + 0.006, y, label, ha='left', va='center',
                fontsize=7, zorder=5, bbox=lbl_box)

    # Draw transitions between levels
    gammas = []
    for ev, next_idx, idx in transition_scheme:
        next_idx = int(next_idx)
        idx = int(idx)
        if next_idx == -1:
            continue

        gamma_e = int(ev)

        x1, y1 = x_pos[idx], to_y(idx)
        x2, y2 = x_pos[next_idx], to_y(next_idx)
        color = colors[idx % len(colors)]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        if gamma_e > 0:
            # Valid transition: coloured arrow + gamma energy label
            ax.annotate("", xy=(x2, y2 + 0.004), xytext=(x1, y1 - 0.004),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                       lw=1.4, mutation_scale=10), zorder=2)
            ax.text(mx + 0.012, my, f"γ = {gamma_e} keV",
                    color=color, fontsize=6, ha='left', va='center',
                    zorder=6, bbox=lbl_box)
            gammas.append(gamma_e)
        else:
            # Invalid transition: dashed grey line + "✗ invalid" label
            ax.plot([x1, x2], [y1, y2], color='#aaaaaa', linewidth=1.0,
                    linestyle='--', zorder=2)
            ax.text(mx, my, f"✗  {gamma_e} keV (invalid)",
                    color='#cc0000', fontsize=6, ha='center', va='center',
                    zorder=6, bbox=inv_lbl_box)

    ax.set_xlim(-0.05, 1.4)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(fig_name, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{fig_name}.png"), dpi=150, bbox_inches='tight')
    print(f"Saved → {fig_name}.png")