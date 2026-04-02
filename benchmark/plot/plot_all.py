from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# === 根目录 ===
root = Path("./")

# === 找两个 rotation level ===
r1_dir = root / "R_1"
r2_dir = root / "R_2"

def find_latest_npz(folder, pattern):
    files = list(folder.rglob(pattern))
    if not files:
        raise RuntimeError(f"No file found in {folder} with pattern {pattern}")
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1]

# === R1 ===
eps_file_r1 = find_latest_npz(r1_dir, "epsilon_sweep_summary_*.npz")
bandit_file_r1 = find_latest_npz(r1_dir, "bandit*_summary_*.npz")

# === R2 ===
eps_file_r2 = find_latest_npz(r2_dir, "epsilon_sweep_summary_*.npz")
bandit_file_r2 = find_latest_npz(r2_dir, "bandit*_summary_*.npz")

print("R1 epsilon:", eps_file_r1)
print("R1 bandit:", bandit_file_r1)
print("R2 epsilon:", eps_file_r2)
print("R2 bandit:", bandit_file_r2)

# === load ===
data_eps_r1 = np.load(eps_file_r1)
data_bandit_r1 = np.load(bandit_file_r1)

data_eps_r2 = np.load(eps_file_r2)
data_bandit_r2 = np.load(bandit_file_r2)

# === 提取数据 ===
def extract(data):
    eps = data["epsilon"]
    cap = data["avg_selected_capacity"]
    idx = np.argsort(eps)
    return eps[idx], cap[idx]

eps1, cap1 = extract(data_eps_r1)
eps2, cap2 = extract(data_bandit_r1)

eps3, cap3 = extract(data_eps_r2)
eps4, cap4 = extract(data_bandit_r2)

# ===============================
# 📊 图1：epsilon vs capacity（四条线）
# ===============================
plt.figure(figsize=(8,6))

plt.plot(eps1, cap1, marker='o', label="R1 epsilon-greedy")
plt.plot(eps2, cap2, marker='s', label="R1 bandit")

plt.plot(eps3, cap3, marker='o', linestyle='--', label="R2 epsilon-greedy")
plt.plot(eps4, cap4, marker='s', linestyle='--', label="R2 bandit")

plt.xlabel("epsilon")
plt.ylabel("Average selected capacity")
plt.title("epsilon vs capacity (R1 vs R2)")
plt.xscale("log")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("compare_epsilon_R1_R2.png", dpi=200)
plt.close()

# ===============================
# 🔥 图2：1/epsilon vs capacity
# ===============================
plt.figure(figsize=(8,6))

plt.plot(1/eps1, cap1, marker='o', label="Rotation level 1: epsilon-greedy")
plt.plot(1/eps2, cap2, marker='s', label="Rotation level 1: bandit")

plt.plot(1/eps3, cap3, marker='o', linestyle='--', label="Rotation level 2: epsilon-greedy")
plt.plot(1/eps4, cap4, marker='s', linestyle='--', label="Rotation level 2: bandit")

plt.xlabel("1 / epsilon")
plt.ylabel("Average selected capacity")
plt.title("1/epsilon vs capacity (R1 vs R2)")
plt.xscale("log")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("compare_inv_epsilon_R1_R2.png", dpi=200)
plt.show()