from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 根目录
# =========================================================
root = Path("./")

# =========================================================
# 工具函数
# =========================================================
def find_latest_npz(folder, pattern):
    files = list(folder.rglob(pattern))
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1]

def extract_curve(npz_path):
    data = np.load(npz_path)

    eps = data["epsilon"]
    cap = data["avg_selected_capacity"]

    idx = np.argsort(eps)
    return eps[idx], cap[idx]

def parse_alpha(folder_name):
    m = re.search(r"_alpha_(.+?)_\d{8}_\d{6}$", folder_name)
    if not m:
        return "unknown"
    return m.group(1)

def alpha_sort_key(alpha_str):
    if alpha_str == "None":
        return (0, -1)
    try:
        return (1, float(alpha_str))
    except:
        return (2, alpha_str)

# =========================================================
# 1️⃣ epsilon-greedy baseline（全局找）
# =========================================================
eps_files = sorted(root.rglob("epsilon_sweep_summary_*.npz"))

if not eps_files:
    print("⚠️ No epsilon baseline found")
    eps_curve = None
else:
    eps_file = sorted(eps_files, key=lambda p: p.stat().st_mtime)[-1]
    print("Using epsilon baseline:", eps_file)
    eps_curve = extract_curve(eps_file)

# =========================================================
# 2️⃣ 找所有 alpha bandit folder
# =========================================================
bandit_dirs = [
    p for p in root.iterdir()
    if p.is_dir() and p.name.startswith("bandit_link_semi_bandit_R_2_alpha_")
]

if not bandit_dirs:
    raise RuntimeError("No bandit alpha folders found")

# =========================================================
# 3️⃣ 加载所有 bandit 曲线
# =========================================================
curves = []

for folder in bandit_dirs:
    alpha = parse_alpha(folder.name)
    npz_file = find_latest_npz(folder, "*summary*.npz")

    if npz_file is None:
        print(f"Skip {folder}, no summary")
        continue

    eps, cap = extract_curve(npz_file)

    curves.append({
        "alpha": alpha,
        "epsilon": eps,
        "capacity": cap,
    })

# 排序
curves = sorted(curves, key=lambda x: alpha_sort_key(x["alpha"]))

# =========================================================
# 📊 图1：epsilon vs capacity
# =========================================================
plt.figure(figsize=(8,6))

# --- baseline ---
if eps_curve is not None:
    plt.plot(
        eps_curve[0],
        eps_curve[1],
        marker='o',
        linewidth=3,
        color='black',
        label="epsilon-greedy (baseline)"
    )

# --- bandit curves ---
for c in curves:
    ls = '--' if c["alpha"] == "None" else '-'
    plt.plot(
        c["epsilon"],
        c["capacity"],
        marker='o',
        linestyle=ls,
        label=f"bandit α={c['alpha']}"
    )

plt.xlabel("epsilon")
plt.ylabel("Average selected capacity")
plt.title("Bandit vs Epsilon-Greedy (all alpha)")
plt.xscale("log")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("compare_all_alpha_with_baseline.png", dpi=200)
plt.close()

# =========================================================
# 📊 图2：1/epsilon vs capacity
# =========================================================
plt.figure(figsize=(8,6))

# --- baseline ---
if eps_curve is not None:
    eps = eps_curve[0]
    cap = eps_curve[1]
    mask = eps > 0

    plt.plot(
        1/eps[mask],
        cap[mask],
        marker='o',
        linewidth=3,
        color='black',
        label="epsilon-greedy (baseline)"
    )

# --- bandit ---
for c in curves:
    eps = c["epsilon"]
    cap = c["capacity"]

    mask = eps > 0
    if not np.any(mask):
        continue

    ls = '--' if c["alpha"] == "None" else '-'

    plt.plot(
        1/eps[mask],
        cap[mask],
        marker='o',
        linestyle=ls,
        label=f"bandit α={c['alpha']}"
    )

plt.xlabel("1 / epsilon")
plt.ylabel("Average selected capacity")
plt.title("Bandit vs Epsilon-Greedy (1/epsilon view)")
plt.xscale("log")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("compare_all_alpha_with_baseline_inv.png", dpi=200)
plt.show()