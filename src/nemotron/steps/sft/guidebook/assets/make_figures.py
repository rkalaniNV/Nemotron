#!/usr/bin/env python3
"""Regenerate the SFT guidebook figures.

    python3 make_figures.py

Every value plotted here is transcribed from the reviewed ablation tables and is listed in the
`EVIDENCE` block below, so a reader can check a figure against the source report without reading
this code. Do not hand-edit a plotted value inside a PNG: change `EVIDENCE`, rerun, and update the
guidebook text in the same commit.

Scores are MILU / GSM8K-Indic / IndicIFEval / IndiVibe on Nemotron-3-Nano-30B-A3B, reasoning-on,
at each arm's selected checkpoint. Standard errors: MILU 0.4pp (English/Hindi) and 0.76pp
(Malayalam), GSM8K-Indic 1.3pp, IndicIFEval 2.4pp, IndiVibe 4.8pp.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

GREEN, BLACK, GREY, SAGE = '#76B900', '#111111', '#8C8C8C', '#A3C182'
BAND, ROW_A, ROW_B, TXT = '#EAF3DC', '#F2F2F2', '#EAF3DC', '#4D4D4D'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 15, 'axes.edgecolor': GREY,
    'axes.linewidth': 1.0, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})


def frame(ax, ylab=None, xlab=None):
    """House style: horizontal grid only, no top/right spines."""
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(GREY)
    ax.spines['bottom'].set_color(GREY)
    ax.yaxis.grid(True, color='#E0E0E0', linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TXT, length=0)
    if ylab:
        ax.set_ylabel(ylab, color=TXT, fontsize=15)
    if xlab:
        ax.set_xlabel(xlab, color=TXT, fontsize=15)


def title(fig, t, sub=None):
    fig.text(0.055, 0.955, t, fontsize=25, fontweight='bold', color=BLACK, va='top')
    if sub:
        fig.text(0.055, 0.895, sub, fontsize=16, color=TXT, va='top')


def foot(fig, s):
    fig.text(0.975, 0.028, s, fontsize=14, fontweight='bold', color=TXT, ha='right')


def save(fig, name):
    fig.savefig(name, dpi=110, bbox_inches=None)
    plt.close(fig)
    print('wrote', name)


# ----------------------------------------------------------------- 1. decision path
def overview():
    rows = [
        ('01', 'CONTROL', 'Prove it is the data', 'Same recipe, neutral data'),
        ('02', 'VOLUME', 'Saturates early', '91% of the gain at 10% of the data'),
        ('03', 'BASE', 'Pre-RL beats post-RL', 'Post-RL spends the run undoing RL'),
        ('04', 'REPAIR', 'Add back IF data', 'Recovers the standing cost, free'),
        ('05', 'LANGUAGE', 'Measure script fidelity', 'MCQ scoring cannot see it'),
        ('06', 'METHOD', 'Full SFT, not LoRA', 'For knowledge injection'),
    ]
    fig = plt.figure(figsize=(12.6, 9.1))
    title(fig, 'Supervised fine-tuning: a measured decision path',
          'Use the portable rule first; the Indic runs show the evidence and the limits.')
    top, h, gap = 0.815, 0.108, 0.021
    for i, (n, k, v, note) in enumerate(rows):
        y = top - i * (h + gap)
        fig.patches.append(Rectangle((0.055, y - h), 0.925, h, transform=fig.transFigure,
                                     facecolor=ROW_B if i % 2 else ROW_A, edgecolor='none'))
        fig.patches.append(Rectangle((0.055, y - h), 0.008, h, transform=fig.transFigure,
                                     facecolor=GREEN, edgecolor='none'))
        fig.text(0.088, y - h / 2, n, fontsize=17, fontweight='bold', color=GREEN, va='center')
        fig.text(0.135, y - h / 2, k, fontsize=16, fontweight='bold', color=BLACK, va='center')
        fig.text(0.315, y - h / 2, v, fontsize=18, fontweight='bold', color=BLACK, va='center')
        fig.text(0.610, y - h / 2, note, fontsize=15, color=TXT, va='center')
    save(fig, 'guidebook_overview.png')


# ----------------------------------------------------------------- 2. data volume
def saturation():
    # boxed50_data_scale, released SFT+RL init, reasoning-on, Hindi MILU loose.
    packs = ['20k', '50k', '80k', '100k', '200k']
    milu = [77.85, 78.13, 77.62, 77.82, 78.42]
    base = 72.02
    gain = [m - base for m in milu]
    fig, ax = plt.subplots(figsize=(12.2, 8.0))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.80, bottom=0.155)
    ax.axvspan(-0.45, 0.45, color=BAND, zorder=0)
    ax.plot(packs, gain, color=GREEN, lw=3.4, marker='o', ms=13, zorder=3)
    ax.axhline(gain[-1], color=GREY, lw=1.6, ls=(0, (7, 4)), zorder=2)
    ax.text(4.06, gain[-1] + 0.06, 'best observed (200k)', color=TXT, fontsize=14,
            ha='right', va='bottom')
    for x, g in zip(packs, gain):
        ax.annotate(f'+{g:.2f}', (x, g), textcoords='offset points', xytext=(0, 15),
                    ha='center', fontsize=15, fontweight='bold', color=BLACK)
    frame(ax, 'Hindi MILU gain over the base (pp)', 'total cultural-MCQ samples (50/50 en:hi)')
    ax.set_ylim(0, 7.6)
    title(fig, 'Cultural-MCQ data saturates an order of magnitude early',
          'Hindi MILU, reasoning-on, released SFT+RL base, each arm at its selected checkpoint.')
    foot(fig, '20k reaches 91% of the 200k gain  ·  the remaining 180k buys +0.57pp')
    save(fig, 'data_volume_saturation.png')


# ----------------------------------------------------------------- 3. IF repair
def if_repair():
    # ifeval_data_effect, reasoning-on, IndicIFEval prompt-level loose.
    groups = ['English\npre-RL base', 'Hindi\npre-RL base',
              'English\nreleased base', 'Hindi\nreleased base']
    base = [82.24, 72.65, 94.69, 84.08]
    mcq = [78.16, 67.14, 82.45, 72.24]
    both = [81.43, 72.04, 86.33, 71.02]
    x = range(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.795, bottom=0.155)
    ax.bar([i - w for i in x], base, w, color=GREY, label='base (no SFT)', zorder=3)
    ax.bar(list(x), mcq, w, color=BLACK, label='+ cultural MCQ 100k', zorder=3)
    ax.bar([i + w for i in x], both, w, color=GREEN, label='+ MCQ 100k + IF 20k en', zorder=3)
    for i, (b, m, t) in enumerate(zip(base, mcq, both)):
        ax.annotate(f'{t - m:+.1f}', (i + w, t), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=14, fontweight='bold', color=GREEN)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    frame(ax, 'IndicIFEval prompt-level loose (%)')
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, loc='upper left', fontsize=15, ncol=3,
              bbox_to_anchor=(0.0, 1.005))
    title(fig, 'The instruction-following cost is real, and repairable',
          'Adding 20k English IF samples to the pack, reasoning-on. Green label = recovery.')
    foot(fig, 'pre-RL recovers fully  ·  released recovers English only  ·  MILU and GSM8K unmoved')
    save(fig, 'instruction_following_repair.png')


# ----------------------------------------------------------------- 4. fidelity blind spot
def fidelity():
    # updesh_translation_hindi + lora_vs_sft_hindi, pre-RL init, reasoning-on.
    # accuracy = Hindi MILU loose; fidelity = share of Hindi GSM8K answers in Devanagari.
    names = ['pre-RL base', 'LoRA\n(MCQ 100k)', 'full SFT\n(MCQ 100k)',
             'full SFT\n(MCQ + translation)']
    milu = [69.48, 72.18, 78.11, 77.97]
    fid = [16.11, 34.70, 90.97, 93.23]
    x = range(len(names))
    w = 0.34
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    fig.subplots_adjust(left=0.09, right=0.975, top=0.795, bottom=0.16)
    ax.bar([i - w / 2 for i in x], milu, w, color=BLACK, label='Hindi MILU (what you measure)',
           zorder=3)
    ax.bar([i + w / 2 for i in x], fid, w, color=GREEN,
           label='Hindi GSM8K answers actually written in Hindi', zorder=3)
    for i, (a, f) in enumerate(zip(milu, fid)):
        ax.annotate(f'{a:.1f}', (i - w / 2, a), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=14, fontweight='bold', color=BLACK)
        ax.annotate(f'{f:.1f}', (i + w / 2, f), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=14, fontweight='bold', color=GREEN)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    frame(ax, 'score / share (%)')
    ax.set_ylim(0, 118)
    ax.legend(frameon=False, loc='upper left', fontsize=15, bbox_to_anchor=(0.0, 1.005))
    title(fig, 'The failure a multiple-choice metric cannot see',
          'Both bars are the same model. MILU grades one option letter; the prose has a language.')
    foot(fig, 'the base answers 84% of Hindi maths questions in English and MILU never shows it')
    save(fig, 'language_fidelity_blindspot.png')


# ----------------------------------------------------------------- 5. generic data
def generic_split():
    # mathstem_lang_split + generic_sft_control, pre-RL init, reasoning-on.
    names = ['English-only\n47k', 'Hindi-only\n47k', 'both\n47k+47k', 'earlier 162k\n+ code']
    gsm = [90.45, 70.58, 79.83, 80.82]
    ife = [73.88, 60.41, 59.18, 52.45]
    gsm_b, ife_b = 89.08, 72.65
    x = range(len(names))
    w = 0.34
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    fig.subplots_adjust(left=0.09, right=0.975, top=0.795, bottom=0.16)
    ax.axvspan(-0.5, 0.5, color=BAND, zorder=0)
    ax.bar([i - w / 2 for i in x], gsm, w, color=BLACK, label='Hindi GSM8K-Indic', zorder=3)
    ax.bar([i + w / 2 for i in x], ife, w, color=GREEN, label='Hindi IndicIFEval', zorder=3)
    ax.axhline(gsm_b, color=BLACK, lw=1.5, ls=(0, (7, 4)), zorder=2)
    ax.axhline(ife_b, color=GREEN, lw=1.5, ls=(0, (7, 4)), zorder=2)
    ax.text(3.48, gsm_b + 0.8, 'base GSM8K', color=BLACK, fontsize=13, ha='right')
    ax.text(3.48, ife_b + 0.8, 'base IndicIFEval', color=GREEN, fontsize=13, ha='right')
    for i, (g, f) in enumerate(zip(gsm, ife)):
        ax.annotate(f'{g - gsm_b:+.1f}', (i - w / 2, g), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=14, fontweight='bold', color=BLACK)
        ax.annotate(f'{f - ife_b:+.1f}', (i + w / 2, f), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=14, fontweight='bold', color=GREEN)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    frame(ax, 'score (%)', 'generic maths/STEM pack, split by the language of its data')
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, loc='upper left', fontsize=15, ncol=2, bbox_to_anchor=(0.0, 1.005))
    title(fig, 'Damage is attributable to one half of a pack',
          'Splitting a harmful generic pack by language, pre-RL base, reasoning-on.')
    foot(fig, 'the English half is clean and reusable  ·  the Hindi half carries all the damage')
    save(fig, 'generic_data_language_split.png')


# ----------------------------------------------------------------- 6. LoRA vs SFT
def lora():
    # lora_vs_sft_hindi / _malayalam, pre-RL init, reasoning-on, at selected checkpoints.
    labels = ['Hindi MILU\ngain', 'Malayalam MILU\ngain', 'English IndicIFEval\nvs base']
    sft = [78.11 - 69.48, 69.98 - 54.48, 76.33 - 82.24]
    lo = [72.18 - 69.48, 58.67 - 54.48, 84.49 - 82.24]
    x = range(len(labels))
    w = 0.34
    fig, ax = plt.subplots(figsize=(13.0, 8.0))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.795, bottom=0.17)
    ax.bar([i - w / 2 for i in x], sft, w, color=BLACK, label='full-parameter SFT (LR 1e-5)',
           zorder=3)
    ax.bar([i + w / 2 for i in x], lo, w, color=GREEN, label='LoRA adapter (LR 1e-4)', zorder=3)
    ax.axhline(0, color=GREY, lw=1.4, zorder=2)
    for i, (a, b) in enumerate(zip(sft, lo)):
        for xx, v, c in ((i - w / 2, a, BLACK), (i + w / 2, b, GREEN)):
            ax.annotate(f'{v:+.2f}', (xx, v), textcoords='offset points',
                        xytext=(0, 8 if v >= 0 else -20), ha='center', fontsize=15,
                        fontweight='bold', color=c)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    frame(ax, 'change from the same pre-RL base (pp)')
    ax.set_ylim(-9, 19)
    ax.legend(frameon=False, loc='upper right', fontsize=15)
    title(fig, 'LoRA is the safer edit and the weaker one',
          'Identical pack, identical base, identical iterations — only the update rule differs.')
    foot(fig, 'LoRA keeps instruction-following  ·  and installs a third of the knowledge')
    save(fig, 'lora_vs_sft.png')


for f in (overview, saturation, if_repair, fidelity, generic_split, lora):
    f()
