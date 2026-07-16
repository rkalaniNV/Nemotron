#!/usr/bin/env python3
"""Render the agentic-RAG pipeline flow diagram -> assets/pipeline_flow.png.

Compact landscape 3x2 grid of equal cards with a snake flow + output footer.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams["font.family"] = "DejaVu Sans"

INK = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"
ARROW = "#94A3B8"
BG = "#FFFFFF"

fig, ax = plt.subplots(figsize=(12.8, 6.8), dpi=260)
ax.set_xlim(0, 100)
ax.set_ylim(0, 53)
ax.axis("off")
fig.patch.set_facecolor(BG)


def _round(x, y, w, h, r, **kw):
    return FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", **kw)


def shadow(x, y, w, h, r=1.4):
    for i, a in enumerate([0.05, 0.045]):
        ax.add_patch(_round(x + 0.2, y - (0.35 + i * 0.3), w, h, r, linewidth=0,
                            facecolor="#0F172A", alpha=a, zorder=1))


def card(x, y, w, h, accent, num, title, sub):
    shadow(x, y, w, h)
    ax.add_patch(_round(x, y, w, h, 1.4, linewidth=1.1, edgecolor=BORDER, facecolor="white", zorder=2))
    ax.add_patch(FancyBboxPatch((x + 1.0, y + h - 1.9), w - 2.0, 1.4,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                linewidth=0, facecolor=accent, zorder=3))  # top accent bar
    ax.add_patch(Circle((x + 3.6, y + h - 4.6), 2.05, facecolor=accent, edgecolor="white",
                        linewidth=1.2, zorder=4))
    ax.text(x + 3.6, y + h - 4.6, str(num), ha="center", va="center", color="white",
            fontsize=10, fontweight="bold", zorder=5)
    ax.text(x + w / 2 + 2.0, y + h - 4.6, title, ha="center", va="center", color=INK,
            fontsize=10.8, fontweight="bold", zorder=5)
    ax.text(x + w / 2, y + 2.6, sub, ha="center", va="center", color=MUTED, fontsize=8.0, zorder=5)


def arrow(x1, y1, x2, y2, color=ARROW, lw=2.0, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=lw, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


# ── title ─────────────────────────────────────────────────────────────────────
ax.text(50, 50.4, "Agentic-RAG  ·  Deep-Research Trajectory SDG", ha="center",
        fontsize=15.5, fontweight="bold", color=INK)
ax.text(50, 46.7, "Stage 1 runs once · Stages 2–6 repeat per cluster  (index built → used → torn down)",
        ha="center", fontsize=9.4, color=MUTED, style="italic")

# ── grid geometry (3 cols x 2 rows) ──────────────────────────────────────────
cw, ch = 27.0, 13.0
col = [6.0, 36.5, 67.0]          # left x of each column
row_top, row_bot = 27.5, 11.5    # bottom y of each row
cxs = [c + cw / 2 for c in col]

stages = [
    ("#3B82F6", "Cluster documents", "embed whole docs · group by topic"),
    ("#10B981", "Generate questions", "2–5 per shard · easy → hard"),
    ("#F59E0B", "Retrieve", "cluster-scoped · 2×k → subsample"),
    ("#F43F5E", "Conversation variant", "single · multi-turn · multi-step"),
    ("#8B5CF6", "Agent tool loop", "search · reason · repeat  (+ compress)"),
    ("#14B8A6", "Evaluate", "LLM judge · validate tool calls"),
]
# snake placement: top row L→R = 1,2,3 ; bottom row L→R = 6,5,4
pos = {0: (col[0], row_top), 1: (col[1], row_top), 2: (col[2], row_top),
       3: (col[2], row_bot), 4: (col[1], row_bot), 5: (col[0], row_bot)}
for i, (accent, title, sub) in enumerate(stages):
    x, y = pos[i]
    card(x, y, cw, ch, accent, i + 1, title, sub)

# ── snake flow arrows ────────────────────────────────────────────────────────
mid_top, mid_bot = row_top + ch / 2, row_bot + ch / 2
arrow(col[0] + cw, mid_top, col[1], mid_top)          # 1 → 2
arrow(col[1] + cw, mid_top, col[2], mid_top)          # 2 → 3
arrow(cxs[2], row_top, cxs[2], row_bot + ch)          # 3 → 4 (down, right side)
arrow(col[2], mid_bot, col[1] + cw, mid_bot)          # 4 → 5 (left)
arrow(col[1], mid_bot, col[0] + cw, mid_bot)          # 5 → 6 (left)

# ── output footer ────────────────────────────────────────────────────────────
arrow(cxs[0], row_bot, cxs[0], 8.6, color="#4338CA")  # 6 → output
shadow(6, 2.0, 88, 6.2, r=1.6)
ax.add_patch(_round(6, 2.0, 88, 6.2, 1.6, linewidth=0, facecolor="#4338CA", zorder=2))
ax.text(30, 5.1, "output/sdg/*.jsonl", ha="center", va="center", color="white",
        fontsize=11.5, fontweight="bold", zorder=3)
ax.text(66, 5.1, "SFT trajectories · messages + tools + metadata (cluster_id · hops · gold_rank · judge)",
        ha="center", va="center", color="#C7D2FE", fontsize=8.0, zorder=3)

plt.savefig("assets/pipeline_flow.png", bbox_inches="tight", facecolor=BG, dpi=260, pad_inches=0.2)
print("wrote assets/pipeline_flow.png")
