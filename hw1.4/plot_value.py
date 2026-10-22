"""绘制 Blackjack 最终策略下状态价值 V(s) 的三维线框图（Sutton 书 Figure 5.1 风格）。

用法：python3 plot_value.py [训练程序输出文件]
默认读取 /tmp/blackjack_out.txt，解析其中 print_value 的最后一组价值表
（11 行 × 每行 20 个浮点数：前 10 个无可用A，后 10 个有可用A），
输出 blackjack_value.png。
"""

import re

import sys

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = "/home/hw/course/rl/hw1.4/blackjack_value.png"
DEALER_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
PLAYER_SUMS = list(range(11, 22))  # 纵轴 11 -> 21

FLOAT_ROW = re.compile(r"^\s*-?\d")


def parse_last_values(path: str) -> tuple[np.ndarray, np.ndarray]:
    """取最后一组价值表，返回 (无可用A, 有可用A) 两个 11x10 数组（行序翻转为 11 -> 21）。"""
    rows = []
    with open(path) as f:
        for line in f:
            if not FLOAT_ROW.match(line):
                continue
            tokens = line.split()
            try:
                nums = [float(t) for t in tokens]
            except ValueError:
                continue
            if len(nums) == 20:  # 价值表：10 + 10 两半边
                rows.append(nums)
    if len(rows) < 11:
        raise SystemExit(f"未找到完整价值表（仅 {len(rows)} 行），请检查输入文件")
    last = np.array(rows[-11:][::-1])  # 最后一组（21 -> 11），翻转为 11 -> 21
    return last[:, :10], last[:, 10:]


def plot(path: str) -> None:
    no_ace, with_ace = parse_last_values(path)
    grids = [no_ace, with_ace]
    titles = ["No usable ace", "Usable ace"]

    X, Y = np.meshgrid(np.arange(10), np.arange(11))
    fig = plt.figure(figsize=(13, 6))
    for i, (grid, title) in enumerate(zip(grids, titles), start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        # 黑白线框：无填充，仅细网格线（Figure 5.1 风格）
        ax.plot_wireframe(X, Y, grid, color="black", linewidth=0.7, rstride=1, cstride=1)
        ax.set_xticks(range(10), DEALER_LABELS)
        ax.set_yticks(range(11), PLAYER_SUMS)
        ax.set_zlim(-1, 1)
        ax.set_zticks([-1, 0, 1], ["-1", "0", "+1"])
        ax.set_box_aspect((4, 4, 1.2))  # 压扁 z 轴，贴近书上扁平的盒子比例
        ax.set_xlabel("Dealer showing", labelpad=8)
        ax.set_ylabel("Player sum", labelpad=8)
        ax.set_title(title)
        # 坐标面背景设为纯白，贴近书上风格
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("white")
            axis.pane.set_edgecolor("lightgray")
        ax.view_init(elev=35, azim=-60)  # Dealer showing 在前、Player sum 在右，与书上 Figure 5.1 一致
    fig.suptitle(r"State-value function $V(s)=Q(s,\pi(s))$ under the learned policy")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    plot(sys.argv[1] if len(sys.argv) > 1 else "/tmp/blackjack_out.txt")
