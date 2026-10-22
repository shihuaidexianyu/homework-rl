"""绘制 Blackjack 最终策略的二维策略图（仿 Sutton 书 Figure 5.2）。

用法：python3 plot_policy.py [训练程序输出文件]
默认读取 /tmp/blackjack_out.txt，解析其中最后一组策略表
（连续 11 行 "HHHHHHHHHH  SSSSSSSSSS" 格式的文本），
输出 blackjack_policy.png。
"""

import re
import sys

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = "/home/hw/course/rl/hw1.4/blackjack_policy.png"
DEALER_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
PLAYER_SUMS = list(range(11, 22))  # 绘图纵轴 11 -> 21（自下而上）


def parse_last_policy(path: str) -> tuple[list[str], list[str]]:
    """从训练输出中取最后一组策略表，返回 (无可用A, 有可用A) 两个 11 行的列表。

    print_policy 按 player_sum = 21 -> 11 打印，这里翻转为 11 -> 21 便于绘图。
    """
    rows = []
    pattern = re.compile(r"^\s*([HS]{10})\s+([HS]{10})\s*$")
    with open(path) as f:
        for line in f:
            m = pattern.match(line)
            if m:
                rows.append((m.group(1), m.group(2)))
    if len(rows) < 11:
        raise SystemExit(f"未找到完整策略表（仅 {len(rows)} 行），请检查输入文件")
    last = rows[-11:]  # 最后一组 11 行 = 最终策略（21 -> 11）
    no_ace = [r[0] for r in reversed(last)]
    with_ace = [r[1] for r in reversed(last)]
    return no_ace, with_ace


def to_grid(rows: list[str]) -> np.ndarray:
    """'H'->0, 'S'->1，转成 11x10 数值网格。"""
    return np.array([[1 if c == "S" else 0 for c in row] for row in rows], dtype=float)


def plot(path: str) -> None:
    no_ace, with_ace = parse_last_policy(path)
    grids = [to_grid(no_ace), to_grid(with_ace)]
    titles = ["No usable ace", "Usable ace"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    cmap = plt.matplotlib.colors.ListedColormap(["#f4a582", "#4393c3"])  # HIT 橙, STICK 蓝
    for ax, grid, title in zip(axes, grids, titles):
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="lower")
        # 单元格文字与网格线
        for y in range(11):
            for x in range(10):
                ax.text(x, y, "S" if grid[y, x] == 1 else "H",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 11, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(10), DEALER_LABELS)
        ax.set_yticks(range(11), PLAYER_SUMS)
        ax.set_xlabel("Dealer showing")
        ax.set_title(title)
    axes[0].set_ylabel("Player sum")

    # 图例
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#f4a582", label="HIT"),
                        Patch(color="#4393c3", label="STICK")],
               loc="upper right")
    fig.suptitle("Blackjack policy learned by Monte Carlo ES (100 rounds)")
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    fig.savefig(OUTPUT, dpi=150)
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    plot(sys.argv[1] if len(sys.argv) > 1 else "/tmp/blackjack_out.txt")
