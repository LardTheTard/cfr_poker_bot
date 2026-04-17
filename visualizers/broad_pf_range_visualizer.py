import plotly.graph_objects as go
import pickle

from pokerkit import Hand
from exact_preflop_freqs import Node

PATH = r'C:\Users\ZhaoLo\poker\cfr_poker_bot\nodesets\preflop_200k.pkl'
RANKS = "AKQJT98765432"
SUITS = "schd"
rank_to_idx = {r: i for i, r in enumerate(RANKS)}

hands = {}

with open(PATH, "rb") as f:
    nodes = pickle.load(f)

def rank_value(card):
    mapping = {
        "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
        "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12,
        "K": 13, "A": 14
    }
    return mapping[card]
        
def bucket_is(hole_cards: str, suited: bool):
    r1 = rank_value(hole_cards[0])
    r2 = rank_value(hole_cards[1])

    high = max(r1, r2)
    low = min(r1, r2)
    pair = (r1 == r2)
    gap = high - low

    # ---------- hand bucket ----------
    if pair:
        if high >= 11:  # JJ+
            hand_bucket = "premium_pairs"
        elif 8 <= high <= 10:  # 88-TT
            hand_bucket = "medium_pairs"
        else:  # 22-77
            hand_bucket = "small_pairs"

    elif suited and high == 14 and 10 <= low <= 13:
        # ATs, AJs, AQs, AKs
        hand_bucket = "premium_suited_aces"

    elif high == 14 and suited and 2 <= low <= 9:
        # A2s-A9s
        hand_bucket = "weak_suited_aces"

    elif high == 14 and not suited and low >= 12:
        # AKo, AQo
        hand_bucket = "premium_offsuit_aces"

    elif suited and high >= 11 and low >= 10:
        # KQs, KJs, QJs, JTs, AQs, AKs etc.
        # Since suited aces already handled above, this is mostly KQs-JTs type hands
        hand_bucket = "premium_suited_broadways"

    elif not suited and high >= 11 and low >= 10:
        # KQo / KJo / QJo / JTo / AQo / AJo / etc.
        if (high == 13 and low == 12):
            hand_bucket = "premium_offsuit_broadways"  # KQo
        elif (high == 14 and low >= 11):
            # AKo/AQo handled above, AJo falls here if you want it weaker
            hand_bucket = "weak_offsuit_broadways"
        elif high == 13 and low == 11:
            hand_bucket = "weak_offsuit_broadways"  # KJo
        elif high == 12 and low == 11:
            hand_bucket = "weak_offsuit_broadways"  # QJo
        elif high == 11 and low == 10:
            hand_bucket = "weak_offsuit_broadways"  # JTo
        else:
            hand_bucket = "trash_offsuit_hands"

    elif suited and gap == 1 and high <= 13:
        # suited connectors like 98s, T9s, 76s, KQs technically connector but already caught above
        hand_bucket = "suited_connectors"

    elif suited and gap == 2 and high <= 13:
        # suited one-gappers like 97s, T8s, J9s
        hand_bucket = "suited_gappers"

    else:
        hand_bucket = "trash_offsuit_hands"

    return hand_bucket

for i in range(13):
    for j in range(i, 13):
        hole_cards = f"{RANKS[i]}{RANKS[j]}"
        bucket = bucket_is(hole_cards, True)
        node = nodes[tuple([bucket, 'BB', 'deep', 'vs_4bet', '~10.0bb raise'])]
        hand = hole_cards + 's'
        node_sum = sum(node.strategy_sum.values())

        if node_sum <= 0:
            print('<= 0 nodesum')

        hands[hand] = {
            "raise": node.strategy_sum.get("raise", 0) / node_sum,
            "check/call": node.strategy_sum.get("check/call", 0) / node_sum,
            "fold": node.strategy_sum.get("fold", 0) / node_sum,
        }
        bucket = bucket_is(hole_cards, False)
        node = nodes[tuple([bucket, 'BB', 'deep', 'root', '~2.0bb raise'])]
        hand = hole_cards + 'o'
        node_sum = sum(node.strategy_sum.values())

        if node_sum <= 0:
            print('<= 0 nodesum')

        hands[hand] = {
            "raise": node.strategy_sum.get("raise", 0) / node_sum,
            "check/call": node.strategy_sum.get("check/call", 0) / node_sum,
            "fold": node.strategy_sum.get("fold", 0) / node_sum,
        }

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

# Colors for each action
ACTION_COLORS = {
    "raise": "#d62728",       # red
    "check/call": "#2ca02c",  # green
    "fold": "#1f77b4",        # blue
}

fig = go.Figure()

# Keep axes in a 13x13 grid
fig.update_xaxes(
    range=[0, 13],
    tickmode="array",
    tickvals=[i + 0.5 for i in range(13)],
    ticktext=list(RANKS),
    side="top",
    showgrid=False,
    zeroline=False,
)

fig.update_yaxes(
    range=[13, 0],  # reversed
    tickmode="array",
    tickvals=[i + 0.5 for i in range(13)],
    ticktext=list(RANKS),
    showgrid=False,
    zeroline=False,
    scaleanchor="x",
    scaleratio=1,
)

# Add cell backgrounds / borders
for i in range(13):
    for j in range(13):
        fig.add_shape(
            type="rect",
            x0=j, x1=j + 1,
            y0=i, y1=i + 1,
            line=dict(color="black", width=1),
            fillcolor="white",
            layer="below",
        )

# Add action-frequency rectangles inside each cell
for hand, freqs in hands.items():
    try:
        r, c = parse_hand(hand)

        x0, x1 = c, c + 1
        y0, y1 = r, r + 1

        # stack actions vertically inside the cell
        # top: raise, middle: call, bottom: fold
        parts = [
            ("raise", freqs.get("raise", 0)),
            ("check/call", freqs.get("check/call", 0)),
            ("fold", freqs.get("fold", 0)),
        ]

        current_y = y0
        for action, frac in parts:
            if frac <= 0:
                continue

            height = frac
            fig.add_shape(
                type="rect",
                x0=x0,
                x1=x1,
                y0=current_y,
                y1=current_y + height,
                line=dict(width=0),
                fillcolor=ACTION_COLORS[action],
            )
            current_y += height

        # Add hover + label
        fig.add_trace(go.Scatter(
            x=[c + 0.5],
            y=[r + 0.5],
            mode="text",
            text=[hand],
            textfont=dict(color="black", size=12),
            hovertemplate=(
                f"<b>{hand}</b><br>"
                f"Raise: {freqs.get('raise', 0):.1%}<br>"
                f"Call: {freqs.get('check/call', 0):.1%}<br>"
                f"Fold: {freqs.get('fold', 0):.1%}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    except Exception as e:
        print(f"Skipping hand {hand}: {e}")

fig.update_layout(
    title="Poker Hand Frequencies",
    width=800,
    height=800,
    plot_bgcolor="white",
    margin=dict(l=40, r=40, t=60, b=40),
)

fig.show()