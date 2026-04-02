"""
CFR+ for Kuhn Poker  —  clean reference implementation
=======================================================

Kuhn Poker:
  Deck: J < Q < K.  Each player antes 1 and gets 1 card.
  History strings use 'p' (pass/check/fold) and 'b' (bet/call).

  Possible terminal histories:
    "pp"  check-check   -> showdown,  pot = 2
    "bp"  bet-fold      -> P1 wins 1  (P2 folded)
    "bb"  bet-call      -> showdown,  pot = 4
    "pbp" chk-bet-fold  -> P1 loses 1 (P1 folded)
    "pbb" chk-bet-call  -> showdown,  pot = 4

CFR+ changes vs. vanilla CFR:
  1. Cumulative regrets clamped to >= 0 after each update.
  2. Strategy sum weighted by iteration t  (more recent = more weight).

Sign convention in this code:
  - All recursive returns are from the CURRENT PLAYER's perspective.
  - Each recursive call is negated: util = -cfr(next_state).
  - This naturally handles the zero-sum property.
"""

from itertools import permutations
from tokenize import Double

CARDS   = ["J", "Q", "K"]       # index = card strength
ACTIONS = ["p", "b"]            # p = pass/check/fold, b = bet/call
TERMINAL_HISTORIES = {"pp", "bp", "bb", "pbp", "pbb"}


# ── Node ───────────────────────────────────────────────────────────────────────
class Node:
    def __init__(self) -> None:
        self.regret_sum   = [0.0, 0.0]
        self.strategy_sum = [0.0, 0.0]

    def current_strategy(self) -> list[float]:
        pos   = [max(r, 0.0) for r in self.regret_sum]
        total = sum(pos)
        return [p / total for p in pos] if total > 0 else [0.5, 0.5]

    def average_strategy(self) -> list[float]:
        total = sum(self.strategy_sum) 
        return [s / total for s in self.strategy_sum] if total > 0 else [0.5, 0.5]


# ── Terminal payoff from CURRENT player's perspective ─────────────────────────
def terminal_util(cards, history):
    """
    Payoff to whichever player just 'received' the last action
    (i.e., the player who did NOT take the last action).
    Equivalently: payoff to player  len(history) % 2.
    """
    plays  = len(history)
    player = plays % 2          # player to receive payoff
    opp    = 1 - player

    last_is_pass = history[-1] == "p"
    double_bet   = history[-2:] == "bb"

    if last_is_pass:
        if history == "pp":     # check-check: showdown
            return 1 if cards[player] > cards[opp] else -1
        else:                   # someone folded: current player wins
            return 1
    else:                       # double bet: showdown with larger pot
        return 2 if cards[player] > cards[opp] else -2


# ── CFR+ recursion ─────────────────────────────────────────────────────────────
nodes = {}

def cfr_plus(cards, history, reach_p0, reach_p1, t):
    """
    Returns expected utility for the CURRENT player.

    cards              : (card_p0, card_p1) as integer indices
    history            : action string so far
    reach_p0, reach_p1 : reach probabilities for each player
    t                  : iteration index (for weighted averaging)
    """
    if history in TERMINAL_HISTORIES:
        return terminal_util(cards, history)

    player  = len(history) % 2
    my_card = CARDS[cards[player]]
    infoset     = f"{my_card}:{history}"

    node     = nodes.setdefault(infoset, Node())
    strategy = node.current_strategy()
    my_reach = reach_p0 if player == 0 else reach_p1
    opp_reach = reach_p1 if player == 0 else reach_p0

    # Recurse, negating result because the child returns its own utility
    util = [0.0, 0.0]
    for a in range(2):
        if player == 0:
            util[a] = -cfr_plus(cards, history + ACTIONS[a],
                                 reach_p0 * strategy[a], reach_p1, t)
        else:
            util[a] = -cfr_plus(cards, history + ACTIONS[a],
                                 reach_p0, reach_p1 * strategy[a], t)

    node_util = sum(strategy[a] * util[a] for a in range(2))

    # CFR+ updates
    for a in range(2):
        regret = opp_reach * (util[a] - node_util)
        node.regret_sum[a]   = max(0.0, node.regret_sum[a] + regret)
        node.strategy_sum[a] += t * my_reach * strategy[a]   # weighted avg

    return node_util


# ── Training ───────────────────────────────────────────────────────────────────
def train(iterations=50_000):
    for t in range(1, iterations + 1):
        for deal in permutations(range(3), 2):    # all 6 card deals
            cfr_plus(deal, "", 1.0, 1.0, t)


# ── Output ─────────────────────────────────────────────────────────────────────
ACTION_NAMES = {
    "":   ("Check", "Bet"),
    "p":  ("Check", "Bet"),
    "b":  ("Fold",  "Call"),
    "pb": ("Fold",  "Call"),
}
SITUATION = {
    "":   "acts first      (P1 turn)",
    "p":  "after P1 check  (P2 turn)",
    "b":  "after P1 bet    (P2 turn)",
    "pb": "after chk-bet   (P1 turn)",
}

def print_results():
    print("\n" + "═" * 66)
    print("  CFR+ converged strategy  (50 000 iterations)")
    print("═" * 66)
    print(f"  {'Card  Situation':<38}  {'Act-0':>8}  {'Act-1':>8}")
    print("─" * 66)
    for key in sorted(nodes):
        card, hist = key.split(":")
        avg        = nodes[key].average_strategy()
        a0, a1     = ACTION_NAMES[hist]
        sit        = SITUATION[hist]
        print(f"  {card}  {sit:<34}  {a0}: {avg[0]:.3f}  {a1}: {avg[1]:.3f}")
    print("═" * 66)

    print("""
Analytic Nash equilibrium  (one solution family, alpha in [0, 1/3]):

  Player 1 (acts first):
    J  ->  Bet alpha         (bluff occasionally)
    Q  ->  Always Check
    K  ->  Bet 3*alpha       (value bet)

  Player 1 (facing check-bet):
    J  ->  Always Fold
    K  ->  Always Call

  Player 2 (facing P1's bet):
    J  ->  Always Fold
    Q  ->  Call 1/3,  Fold 2/3
    K  ->  Always Call

  Player 2 (after P1 checks):
    J  ->  Always Check
    Q  ->  Bet 1/3
    K  ->  Always Bet
""")

if __name__ == "__main__":
    ITERS = 50_000
    print(f"Training CFR+ on Kuhn Poker for {ITERS} iterations ...")
    train(ITERS)
    print_results()
