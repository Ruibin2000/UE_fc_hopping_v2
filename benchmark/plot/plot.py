from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# === 找文件 ===
root = Path("./")

eps_file = sorted(root.glob("epsilon_sweep_summary_*.npz"), key=lambda p: p.stat().st_mtime)[-1]
bandit_file = sorted(root.glob("bandit_link_sweep_summary_*.npz"), key=lambda p: p.stat().st_mtime)[-1]

print("epsilon file:", eps_file)
print("bandit file:", bandit_file)

# === load ===
data_eps = np.load(eps_file)
data_bandit = np.load(bandit_file)

# === 数据 ===
eps1 = data_eps["epsilon"]
cap1 = data_eps["avg_selected_capacity"]

eps2 = data_bandit["epsilon"]
cap2 = data_bandit["avg_selected_capacity"]

# === 排序 ===
idx1 = np.argsort(eps1)
idx2 = np.argsort(eps2)

eps1, cap1 = eps1[idx1], cap1[idx1]
eps2, cap2 = eps2[idx2], cap2[idx2]

# ===============================
# 📊 图1：epsilon vs capacity
# ===============================
plt.figure(figsize=(7,5))

plt.plot(eps1, cap1, marker='o', label="epsilon-greedy")
plt.plot(eps2, cap2, marker='s', label="bandit-RX")

plt.xlabel("epsilon")
plt.ylabel("Average selected capacity")
plt.title("epsilon vs capacity")
plt.grid(True, alpha=0.3)
plt.legend()
plt.xscale("log")

plt.tight_layout()
plt.savefig("compare_epsilon.png", dpi=200)
plt.close()

# ===============================
# 🔥 图2：1/epsilon vs capacity（重点）
# ===============================
x1 = 1.0 / eps1
x2 = 1.0 / eps2

plt.figure(figsize=(7,5))

plt.plot(x1, cap1, marker='o', label="epsilon-greedy")
plt.plot(x2, cap2, marker='s', label="bandit-RX")

plt.xlabel("1 / epsilon")
plt.ylabel("Average selected capacity")
plt.title("1/epsilon vs capacity")

plt.xscale("log")   # ⭐ 强烈推荐
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("compare_inv_epsilon.png", dpi=200)
plt.show()