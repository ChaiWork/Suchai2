import os
import csv
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# --- Professional Dark Theme Configuration ---
DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"
ACCENT_COLORS = ["#00d2ff", "#7b2ff7", "#ff6b6b", "#ffd93d", "#6bcb77",
                 "#ff9a3c", "#ee4266", "#4ecdc4"]

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.4,
    "legend.facecolor": PANEL_BG,
    "legend.edgecolor": GRID_COLOR,
    "legend.fontsize": 9,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica"],
})


def _force_int_xaxis(ax):
    """Force x-axis to show only integer epoch ticks."""
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def smooth(data, weight=0.6):
    """Exponential moving average for smoothing curves."""
    smoothed = []
    clean_data = [float(d) if (d is not None and d != "") else 0.0 for d in data]
    last = clean_data[0] if clean_data else 0.0
    for d in clean_data:
        s = last * weight + d * (1.0 - weight)
        smoothed.append(s)
        last = s
    return smoothed


def read_csv_dict(csv_path):
    """Read a CSV file and return a dict of lists keyed by column name."""
    data = {}
    if not os.path.exists(csv_path):
        return data
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key is None:
                    continue
                if key not in data:
                    data[key] = []
                if val is None or val == "":
                    data[key].append(0.0)
                else:
                    try:
                        data[key].append(float(val))
                    except (ValueError, TypeError):
                        data[key].append(val)
    return data


def plot_training_curves(data, output_dir):
    """Graph 1: Training Curves Dashboard — Win Rate, Loss, Reward (3-panel)."""
    epochs = data.get("epoch", [])
    if not epochs:
        return

    win_rates = data.get("win_rate", [])
    losses = data.get("avg_loss", [])
    rewards = data.get("avg_reward", [])

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Training Curves Dashboard", fontsize=16, fontweight="bold", y=1.02)

    # Win Rate
    axs[0].plot(epochs, win_rates, color=ACCENT_COLORS[0], alpha=0.3, linewidth=1)
    axs[0].plot(epochs, smooth(win_rates), color=ACCENT_COLORS[0], linewidth=2.5, label="Win Rate (smoothed)")
    if win_rates and len(epochs) > 1:
        best_idx = win_rates.index(max(win_rates))
        axs[0].annotate(f"Best: {win_rates[best_idx]:.1f}%",
                        xy=(epochs[best_idx], win_rates[best_idx]),
                        xytext=(10, -20), textcoords="offset points",
                        fontsize=9, color=ACCENT_COLORS[3],
                        arrowprops=dict(arrowstyle="->", color=ACCENT_COLORS[3]))
    axs[0].set_title("Win Rate Progression")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Win Rate (%)")
    axs[0].set_ylim(-5, 105)
    axs[0].grid(True, linestyle="--")
    axs[0].legend()
    _force_int_xaxis(axs[0])

    # Loss
    axs[1].plot(epochs, losses, color=ACCENT_COLORS[2], alpha=0.3, linewidth=1)
    axs[1].plot(epochs, smooth(losses), color=ACCENT_COLORS[2], linewidth=2.5, label="Loss (smoothed)")
    axs[1].set_title("Model Training Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].grid(True, linestyle="--")
    axs[1].legend()
    _force_int_xaxis(axs[1])

    # Reward
    axs[2].plot(epochs, rewards, color=ACCENT_COLORS[4], alpha=0.3, linewidth=1)
    axs[2].plot(epochs, smooth(rewards), color=ACCENT_COLORS[4], linewidth=2.5, label="Avg Reward (smoothed)")
    axs[2].axhline(y=0, color=TEXT_COLOR, linestyle=":", alpha=0.3)
    axs[2].set_title("Episode Reward Curve")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Reward")
    axs[2].grid(True, linestyle="--")
    axs[2].legend()
    _force_int_xaxis(axs[2])

    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_reward_breakdown(data, output_dir):
    """Graph 2: Reward Component Breakdown — Split into step rewards and terminal reward."""
    epochs = data.get("epoch", [])
    if not epochs:
        return

    # Step-level components (small scale)
    step_components = {
        "Prize Taken": ("r_prize_taken", ACCENT_COLORS[4]),
        "Prize Lost": ("r_prize_lost", ACCENT_COLORS[2]),
        "KOs": ("r_kos", ACCENT_COLORS[0]),
        "Own KOs": ("r_own_kos", ACCENT_COLORS[5]),
        "Energy": ("r_energy", ACCENT_COLORS[3]),
        "Bench": ("r_bench", ACCENT_COLORS[1]),
        "Deck-Out": ("r_deckout", ACCENT_COLORS[6]),
        "Stall": ("r_stall", "#4ecdc4"),
        "No Energy": ("r_no_energy", "#a29bfe"),
        "Strategic": ("r_strategic", "#fd79a8"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("Reward Component Breakdown (Per Step Average)", fontsize=15, fontweight="bold", y=1.02)

    # Left panel: Step rewards (grouped bar chart)
    x = np.arange(len(epochs))
    n_components = len(step_components)
    bar_width = 0.8 / n_components

    for i, (label, (key, color)) in enumerate(step_components.items()):
        vals = np.array(data.get(key, [0.0] * len(epochs)), dtype=float)
        offset = (i - n_components / 2 + 0.5) * bar_width
        bars = ax1.bar(x + offset, vals, bar_width, label=label, color=color, alpha=0.85, edgecolor=GRID_COLOR, linewidth=0.5)
        # Add value annotations for non-zero bars
        for bar, val in zip(bars, vals):
            if abs(val) > 0.01:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
                         fontsize=6, color=TEXT_COLOR)

    ax1.axhline(y=0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5)
    ax1.set_title("Step Rewards (Intermediate Signals)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg Reward per Step")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(int(e)) for e in epochs])
    ax1.grid(True, axis="y", linestyle="--")
    ax1.legend(loc="best", fontsize=7, ncol=2)

    # Right panel: Terminal reward (bar chart, separate scale)
    terminal = np.array(data.get("r_terminal", [0.0] * len(epochs)), dtype=float)
    colors_terminal = [ACCENT_COLORS[4] if v >= 0 else ACCENT_COLORS[2] for v in terminal]
    bars = ax2.bar(x, terminal, 0.6, color=colors_terminal, alpha=0.85, edgecolor=GRID_COLOR, linewidth=0.5)
    for bar, val in zip(bars, terminal):
        offset = 0.005 if val >= 0 else -0.005
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + offset,
                 f"{val:.3f}", ha="center", va="bottom" if val >= 0 else "top",
                 fontsize=9, fontweight="bold", color=TEXT_COLOR)
    ax2.axhline(y=0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5)
    ax2.set_title("Terminal Reward (Win/Loss)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Avg Terminal Reward")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(e)) for e in epochs])
    ax2.grid(True, axis="y", linestyle="--")

    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(os.path.join(output_dir, "reward_breakdown.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_deck_matchup(deck_data, output_dir):
    """Graph 3: Deck Matchup Heatmap — epochs × opponent decks."""
    if not deck_data or "epoch" not in deck_data:
        return

    # Build heatmap data
    epochs_raw = deck_data.get("epoch", [])
    names = deck_data.get("deck_name", [])
    win_rates = deck_data.get("win_rate", [])

    if not epochs_raw or not names:
        return

    unique_epochs = sorted(set(float(e) for e in epochs_raw))
    unique_decks = sorted(set(str(n) for n in names))

    if not unique_decks or not unique_epochs:
        return

    heatmap = np.full((len(unique_decks), len(unique_epochs)), np.nan)
    deck_idx = {d: i for i, d in enumerate(unique_decks)}
    epoch_idx = {e: i for i, e in enumerate(unique_epochs)}

    for e, n, wr in zip(epochs_raw, names, win_rates):
        di = deck_idx.get(str(n))
        ei = epoch_idx.get(float(e))
        if di is not None and ei is not None:
            heatmap[di, ei] = float(wr)

    fig, ax = plt.subplots(figsize=(max(10, len(unique_epochs) * 0.6), max(5, len(unique_decks) * 0.6)))

    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100,
                   interpolation="nearest")

    # Annotate cells with win rate values
    for i in range(len(unique_decks)):
        for j in range(len(unique_epochs)):
            val = heatmap[i, j]
            if not np.isnan(val):
                text_color = "black" if val > 50 else "white"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(unique_epochs)))
    ax.set_xticklabels([str(int(e)) for e in unique_epochs], fontsize=8)
    ax.set_yticks(range(len(unique_decks)))
    ax.set_yticklabels([d[:20] for d in unique_decks], fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Opponent Deck")
    ax.set_title("Deck Matchup Win Rate Heatmap", fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Win Rate (%)")
    cbar.ax.yaxis.label.set_color(TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)

    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(os.path.join(output_dir, "deck_matchup.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _get_strategic_actions():
    """Return the strategic action types (excluding END which is just turn confirmation)."""
    # END is excluded because it's a normal turn-ending confirmation, not a strategic choice.
    # "other" includes CARD selection, YES/NO confirmations, energy selection, etc.
    action_types = ["attack", "play", "attach", "evolve", "ability", "retreat", "other"]
    action_labels = ["Attack", "Play Card", "Attach Energy", "Evolve", "Ability", "Retreat", "Tactical Select"]
    colors = [ACCENT_COLORS[2], ACCENT_COLORS[0], ACCENT_COLORS[3], ACCENT_COLORS[4],
              ACCENT_COLORS[1], ACCENT_COLORS[5], ACCENT_COLORS[7]]
    return action_types, action_labels, colors


def plot_action_distribution(action_data, output_dir):
    """Graph 4: Action Distribution — Stacked area chart over epochs (excluding END)."""
    if not action_data or "epoch" not in action_data:
        return

    epochs = [int(e) for e in action_data.get("epoch", [])]
    if not epochs:
        return

    action_types, action_labels, colors = _get_strategic_actions()

    # Convert raw counts to percentages (END excluded from total)
    raw = {}
    for at in action_types:
        raw[at] = np.array([float(x) for x in action_data.get(at, [0] * len(epochs))])

    totals = sum(raw[at] for at in action_types)
    totals = np.maximum(totals, 1)  # Avoid division by zero

    percentages = {}
    for at in action_types:
        percentages[at] = (raw[at] / totals) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2, 1]})

    # Left: Stacked area chart
    y_stack = np.row_stack([percentages[at] for at in action_types])
    ax1.stackplot(epochs, y_stack, labels=action_labels, colors=colors, alpha=0.85)
    ax1.set_title("Strategic Action Distribution Over Training", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Action Proportion (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis="y", linestyle="--")
    ax1.legend(loc="upper right", fontsize=8)
    _force_int_xaxis(ax1)

    # Right: Pie chart of latest epoch (excluding END)
    last_vals = [float(percentages[at][-1]) if len(percentages[at]) > 0 else 0 for at in action_types]
    # Filter out tiny slices (< 0.5%)
    filtered = [(l, v, c) for l, v, c in zip(action_labels, last_vals, colors) if v > 0.5]
    if filtered:
        labels_f, vals_f, colors_f = zip(*filtered)
        wedges, texts, autotexts = ax2.pie(
            vals_f, labels=labels_f, colors=colors_f,
            autopct="%1.0f%%", startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor=DARK_BG, linewidth=1.5),
            textprops={"fontsize": 10, "color": TEXT_COLOR})
        for t in autotexts:
            t.set_fontweight("bold")
            t.set_fontsize(11)
    ax2.set_title(f"Epoch {epochs[-1]} — Strategic Decisions", fontsize=13, fontweight="bold")

    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(os.path.join(output_dir, "action_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_selfplay_progress(data, output_dir):
    """Graph 5: Self-Play Improvement — Win rate + reward over epochs."""
    epochs = data.get("epoch", [])
    if not epochs or len(epochs) < 2:
        return

    win_rates = data.get("win_rate", [])
    rewards = data.get("avg_reward", [])

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_wr = ACCENT_COLORS[0]
    color_rw = ACCENT_COLORS[4]

    # Win rate line
    ax1.plot(epochs, smooth(win_rates), color=color_wr, linewidth=2.5, label="Win Rate (%)", marker="o", markersize=4)
    ax1.fill_between(epochs, 0, smooth(win_rates), alpha=0.1, color=color_wr)
    ax1.set_xlabel("Epoch (Checkpoint)")
    ax1.set_ylabel("Win Rate (%)", color=color_wr)
    ax1.tick_params(axis="y", labelcolor=color_wr)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--")
    _force_int_xaxis(ax1)

    # Reward on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, smooth(rewards), color=color_rw, linewidth=2.0, linestyle="--", label="Avg Reward", marker="d", markersize=4)
    ax2.set_ylabel("Average Reward", color=color_rw)
    ax2.tick_params(axis="y", labelcolor=color_rw)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title("Self-Play Improvement Over Training", fontsize=14, fontweight="bold")
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(os.path.join(output_dir, "selfplay_progress.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_strategy_report(data, deck_data, action_data, output_dir):
    """Graph 6: Combined 2×3 Strategy Report Page."""
    epochs = data.get("epoch", [])
    if not epochs:
        return

    win_rates = data.get("win_rate", [])
    losses = data.get("avg_loss", [])
    rewards = data.get("avg_reward", [])

    fig, axs = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle("Pokémon TCG AI — Strategy Report", fontsize=20, fontweight="bold", y=0.98,
                 color=ACCENT_COLORS[0])

    # (0,0) Win Rate
    axs[0, 0].plot(epochs, win_rates, color=ACCENT_COLORS[0], alpha=0.3, linewidth=1)
    axs[0, 0].plot(epochs, smooth(win_rates), color=ACCENT_COLORS[0], linewidth=2.5)
    axs[0, 0].set_title("Win Rate Progression")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Win Rate (%)")
    axs[0, 0].set_ylim(-5, 105)
    axs[0, 0].grid(True, linestyle="--")
    _force_int_xaxis(axs[0, 0])

    # (0,1) Loss
    axs[0, 1].plot(epochs, losses, color=ACCENT_COLORS[2], alpha=0.3, linewidth=1)
    axs[0, 1].plot(epochs, smooth(losses), color=ACCENT_COLORS[2], linewidth=2.5)
    axs[0, 1].set_title("Training Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].grid(True, linestyle="--")
    _force_int_xaxis(axs[0, 1])

    # (0,2) Reward
    axs[0, 2].plot(epochs, rewards, color=ACCENT_COLORS[4], alpha=0.3, linewidth=1)
    axs[0, 2].plot(epochs, smooth(rewards), color=ACCENT_COLORS[4], linewidth=2.5)
    axs[0, 2].axhline(y=0, color=TEXT_COLOR, linestyle=":", alpha=0.3)
    axs[0, 2].set_title("Reward Curve")
    axs[0, 2].set_xlabel("Epoch")
    axs[0, 2].set_ylabel("Reward")
    axs[0, 2].grid(True, linestyle="--")
    _force_int_xaxis(axs[0, 2])

    # (1,0) Reward Components — Step rewards on left axis, Terminal on right axis
    step_components = [
        ("r_prize_taken", "Prize+", ACCENT_COLORS[4]),
        ("r_prize_lost", "Prize-", ACCENT_COLORS[2]),
        ("r_kos", "KOs+", ACCENT_COLORS[0]),
        ("r_own_kos", "KOs-", ACCENT_COLORS[5]),
        ("r_energy", "Energy", ACCENT_COLORS[3]),
        ("r_bench", "Bench", ACCENT_COLORS[6]),
        ("r_strategic", "Strategic", "#fd79a8"),
        ("r_stall", "Stall", "#4ecdc4"),
        ("r_no_energy", "No Energy", "#a29bfe"),
    ]
    for key, label, color in step_components:
        vals = data.get(key, [])
        if vals:
            axs[1, 0].plot(epochs, smooth(vals), linewidth=1.8, label=label, color=color, marker=".", markersize=3)
    axs[1, 0].axhline(y=0, color=TEXT_COLOR, linestyle=":", alpha=0.3)
    axs[1, 0].set_title("Step Reward Components")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Avg Step Reward")
    axs[1, 0].grid(True, linestyle="--")
    axs[1, 0].legend(fontsize=7, ncol=2)
    _force_int_xaxis(axs[1, 0])

    # Add terminal on twin axis so it doesn't crush step signals
    terminal = data.get("r_terminal", [])
    if terminal:
        ax_twin = axs[1, 0].twinx()
        ax_twin.plot(epochs, smooth(terminal), linewidth=2.0, label="Terminal", color=ACCENT_COLORS[1],
                     linestyle="--", alpha=0.7)
        ax_twin.set_ylabel("Terminal Reward", color=ACCENT_COLORS[1], fontsize=9)
        ax_twin.tick_params(axis="y", labelcolor=ACCENT_COLORS[1])
        ax_twin.legend(fontsize=7, loc="lower left")

    # (1,1) Action Distribution pie (excluding END)
    if action_data and "epoch" in action_data:
        action_types, action_labels_list, pie_colors = _get_strategic_actions()
        last_vals = []
        for at in action_types:
            vals = action_data.get(at, [0])
            last_vals.append(float(vals[-1]) if vals else 0)
        total = sum(last_vals)
        if total > 0:
            filtered = [(l, v, c) for l, v, c in zip(action_labels_list, last_vals, pie_colors) if v > 0]
            if filtered:
                labels_f, vals_f, colors_f = zip(*filtered)
                wedges, texts, autotexts = axs[1, 1].pie(
                    vals_f, labels=labels_f, colors=colors_f, autopct="%1.0f%%",
                    startangle=90, pctdistance=0.75,
                    wedgeprops=dict(edgecolor=DARK_BG, linewidth=1.5),
                    textprops={"fontsize": 9, "color": TEXT_COLOR})
                for t in autotexts:
                    t.set_fontweight("bold")
        ep_label = int(action_data["epoch"][-1]) if action_data.get("epoch") else "?"
        axs[1, 1].set_title(f"Strategic Actions (Epoch {ep_label})")
    else:
        axs[1, 1].text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=12)
        axs[1, 1].set_title("Action Distribution")

    # (1,2) Game Length Over Epochs
    game_lengths = data.get("avg_game_length", [])
    if game_lengths:
        axs[1, 2].plot(epochs, game_lengths, color=ACCENT_COLORS[5], alpha=0.3, linewidth=1)
        axs[1, 2].plot(epochs, smooth(game_lengths), color=ACCENT_COLORS[5], linewidth=2.5)
        axs[1, 2].set_title("Average Game Length")
        axs[1, 2].set_xlabel("Epoch")
        axs[1, 2].set_ylabel("Turns per Game")
        axs[1, 2].grid(True, linestyle="--")
        _force_int_xaxis(axs[1, 2])
    else:
        axs[1, 2].text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=12)
        axs[1, 2].set_title("Game Length")

    try:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    except Exception:
        pass
    plt.savefig(os.path.join(output_dir, "strategy_report.png"), dpi=200, bbox_inches="tight")
    plt.close()


def plot_metrics(csv_path="out/training_metrics.csv", output_dir="out",
                 deck_matchup_csv=None, action_dist_csv=None):
    """Main entry point: reads all CSV files and generates all graphs."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist. Cannot plot metrics.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Read main training metrics
    data = read_csv_dict(csv_path)
    if not data.get("epoch"):
        print("Warning: No epoch data found in training metrics CSV.")
        return

    # Read deck matchup data
    deck_data = read_csv_dict(deck_matchup_csv) if deck_matchup_csv else {}

    # Read action distribution data
    action_data = read_csv_dict(action_dist_csv) if action_dist_csv else {}

    # Generate all graphs
    print("  -> Generating training curves dashboard...")
    plot_training_curves(data, output_dir)

    print("  -> Generating reward component breakdown...")
    plot_reward_breakdown(data, output_dir)

    print("  -> Generating deck matchup heatmap...")
    plot_deck_matchup(deck_data, output_dir)

    print("  -> Generating action distribution chart...")
    plot_action_distribution(action_data, output_dir)

    print("  -> Generating self-play progress chart...")
    plot_selfplay_progress(data, output_dir)

    print("  -> Generating combined strategy report...")
    plot_strategy_report(data, deck_data, action_data, output_dir)

    print(f"Generated plots successfully in '{output_dir}/'")


if __name__ == "__main__":
    plot_metrics()
