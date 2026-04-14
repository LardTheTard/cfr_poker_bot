import plotly.graph_objects as go
import numpy as np
import pickle
from exact_preflop_freqs import Node

PATH = r'C:\Users\ZhaoLo\poker\cfr_poker_bot\nodesets\exact_preflop_1m.pkl'
RANKS = "AKQJT98765432"
rank_to_idx = {r: i for i, r in enumerate(RANKS)}

hands = dict()

with open(PATH, "rb") as f:
    nodes = pickle.load(f)

for bucket, node in nodes.items():
    node_sum = sum(node.strategy_sum.values())
    for k, v in node.strategy_sum.items():
        try:
            hands[bucket[0]] = {k: round(v/node_sum * 100, 1)}
        except ZeroDivisionError:
            pass
    

# hands = {
#     "JQo": {"raise": 0.55, "check/call": 0.30, "fold": 0.15},
#     "3Ts": {"raise": 0.10, "check/call": 0.60, "fold": 0.30},
#     "AKs": {"raise": 0.80, "check/call": 0.15, "fold": 0.05},
#     "55":  {"raise": 0.40, "check/call": 0.50, "fold": 0.10},
# }

def parse_hand(hand):
    """
    Returns (row, col) indices for a 13x13 poker grid
    """
    r1, r2 = hand[0], hand[1]
    i, j = rank_to_idx[r1], rank_to_idx[r2]

    if hand[0] == hand[1]:  # pocket pair
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
        Z[r, c] = freqs["raise"]

        hover_text[r][c] = (
            f"<b>{hand}</b><br>"
            f"Raise: {freqs['raise']:.0%}<br>"
            f"Call: {freqs['check/call']:.0%}<br>"
            f"Fold: {freqs['fold']:.0%}"
        )
    except KeyError:
        pass

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