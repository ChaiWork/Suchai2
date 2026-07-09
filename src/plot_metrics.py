import os
import csv
import matplotlib.pyplot as plt

def plot_metrics(csv_path="out/training_metrics.csv", output_dir="out"):
    """Reads training metrics from CSV and plots curves for Win Rate, Loss, and Reward."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist. Cannot plot metrics.")
        return

    epochs = []
    win_rates = []
    losses = []
    rewards = []

    # Read data from CSV
    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            win_rates.append(float(row["win_rate"]))
            losses.append(float(row["avg_loss"]))
            rewards.append(float(row["avg_reward"]))

    os.makedirs(output_dir, exist_ok=True)

    # 1. Plot Win Rate Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, win_rates, marker="o", color="#2ca02c", linewidth=2.5, label="Win Rate")
    plt.title("Win Rate Curve (vs Random Agent)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Win Rate (%)", fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "win_rate_curve.png"), dpi=150)
    plt.close()

    # 2. Plot Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="s", color="#d62728", linewidth=2.5, label="Loss")
    plt.title("Model Training Loss Curve", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # 3. Plot Episode Reward Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, rewards, marker="d", color="#1f77b4", linewidth=2.5, label="Avg Reward (MCTS Value)")
    plt.title("Episode Reward Curve (Average Root Value)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_curve.png"), dpi=150)
    plt.close()

    print(f"Generated plots successfully in '{output_dir}/'")

if __name__ == "__main__":
    plot_metrics()
