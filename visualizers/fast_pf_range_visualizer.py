import plotly.graph_objects as go
import numpy as np
import pickle

from pokerkit import Hand
from exact_preflop_freqs import Node

PATH = r'C:\Users\login\RANDOM_CODE\wpt_bot\nodesets\nodes_2026-04-16_21-49-33.pkl'
RANKS = "AKQJT98765432"
rank_to_idx = {r: i for i, r in enumerate(RANKS)}

hands = {}

with open(PATH, "rb") as f:
    nodes = pickle.load(f)

for bucket, node in nodes.items():
    if bucket[3] == 'root':
        hand = bucket[0]
        node_sum = sum(node.strategy_sum.values())

        if node_sum <= 0:
            print("Node sum <= 0, strange behaviour.")
            continue

        hands[hand] = {}
        for action, value in node.strategy_sum.items():
            hands[hand][action] = value / node_sum   # keep as 0–1, not 0–100

def parse_hand(hand):
    """
    Returns (row, col) indices for a 13x13 poker grid.
    Pairs go on diagonal.
    Suited go above diagonal.
    Offsuit go below diagonal.
    """
    r1, r2 = hand[0], hand[1]
    i, j = rank_to_idx[r1], rank_to_idx[r2]

    if r1 == r2:
        return i, i
    elif hand[2] == "s":
        return min(i, j), max(i, j)
    else:
        return max(i, j), min(i, j)

Z = np.full((13, 13), np.nan)
hover_text = [["" for _ in range(13)] for _ in range(13)]

for hand, freqs in hands.items():
    try:
        r, c = parse_hand(hand)

        raise_freq = freqs.get("raise", 0.0)
        call_freq = freqs.get("check/call", 0.0)
        fold_freq = freqs.get("fold", 0.0)

        Z[r, c] = raise_freq

        hover_text[r][c] = (
            f"<b>{hand}</b><br>"
            f"Raise: {raise_freq:.1%}<br>"
            f"Call: {call_freq:.1%}<br>"
            f"Fold: {fold_freq:.1%}"
        )
    except Exception as e:
        print(f"Skipping hand {hand}: {e}")

fig = go.Figure(
    data=go.Heatmap(
        z=Z,
        x=list(RANKS),
        y=list(RANKS),
        text=hover_text,
        hoverinfo="text",
        zmin=0,
        zmax=1,
        colorscale=[
            [0.0, "rgb(245,245,245)"],
            [0.25, "rgb(255,200,200)"],
            [0.50, "rgb(255,140,140)"],
            [0.75, "rgb(220,60,60)"],
            [1.0, "rgb(150,0,0)"],
        ],
        colorbar=dict(title="Raise Frequency"),
    )
)

fig.update_layout(
    title="Poker Hand Raise Frequencies",
    width=750,
    height=750,
    xaxis=dict(
        side="top",
        tickangle=0,
        showgrid=False,
    ),
    yaxis=dict(
        autorange="reversed",
        showgrid=False,
    ),
)

fig.show()