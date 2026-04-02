import random
from pokerkit import Automation, Mode, NoLimitTexasHoldem, State
from collections import defaultdict
from logger import Logger
from bucketer import Bucketer

bucketer = Bucketer()

def is_terminal(state):
    return state.actor_index is None

def payoff_p0(state):
    # pfa bets 1/3 pot
    # calls
    # check to showdown
    return 

# ── Info-set node ──────────────────────────────────────────────────────────────

class Node:
    def __init__(self):
        self.regret_sum   = defaultdict(float)
        self.strategy_sum = defaultdict(float)

    def get_strategy(self, actions):
        """Regret-matching (no reach weighting needed for external sampling)."""
        pos = sum(max(self.regret_sum[a], 0.0) for a in actions)
        if pos > 0:
            return {a: max(self.regret_sum[a], 0.0) / pos for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    def accumulate_strategy(self, strat, actions):
        """Update average strategy (called for traverser only)."""
        for action in actions:
            self.strategy_sum[action] += strat[action]

    def avg_strategy(self, actions):
        total = sum(self.strategy_sum[a] for a in actions)
        if total > 0:
            return {a: self.strategy_sum[a] / total for a in actions}
        return {a: 1.0 / len(actions) for a in actions}


# ── External sampling MCCFR ────────────────────────────────────────────────────

nodes = {}

def mccfr(cards, state: State, traverser):
    """
    """
    bucket = bucketer.preflop_bucket(state, pf_history)

    if is_terminal(state):
        v = payoff_p0(state, cards)
        return v if traverser == 0 else -v

    p       = state.actor_index
    actions = ['fold', 'check/call', 'raise'] # IMPLEMENT DIFFERENT RAISE SIZES: 'min_click', 'raise_medium', 'raise_big'

    if bucket not in nodes:
        nodes[bucket] = Node()
    node  = nodes[bucket]
    strat = node.get_strategy(actions)

    if p == traverser:
        # ── Traversing player: explore every action ──────────────────────────
        node.accumulate_strategy(strat, actions)   # track average strategy

        vals = {}

        for action in actions:
            next_state = state
            match action:
                case 'fold':
                    next_state.fold()
                case 'check/call':
                    next_state.check_or_call()
                case 2:
                    if 'vs_4bet' in bucket:
                        amount = max(next_state.stacks)
                    else:
                        amount = next_state.pots[-1] * 1/2 #Raises half pot by default
                    next_state.complete_bet_or_raise_to(amount)
                    next_state.push
            vals[action] = mccfr(cards, next_state, traverser)

        v    = sum(strat[a] * vals[a] for a in actions)

        # Instantaneous counterfactual regret (no reach prob needed here)
        for a in actions:
            node.regret_sum[a] += vals[a] - v

        return v

    else:
        # ── Opponent: SAMPLE a single action ─────────────────────────────────
        probs = [strat[a] for a in actions]
        sampled_a = random.choices(actions, weights=probs)[0]
        return mccfr(cards, h + sampled_a, traverser)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(iters=100_000):
    """Two traversals per iteration (alternate which player is traverser)."""
    p0_total = 0.0
    bucketer = Bucketer()
    for _ in range(iters):
        state = create_state()
        v0 = play_hand(state, traverser=0)
        v1 = play_hand(state, traverser=1)
        p0_total += v0

    print(f"\nTraining complete ({iters:,} iterations, {2*iters:,} traversals)")
    print(f"  Avg P0 value (from P0 traversals) : {p0_total/iters:+.5f}")
    print(f"  Nash equilibrium value            : {-1/18:+.5f}\n")

def play_hand(state, traverser):
    bucket = bucketer.preflop_bucket(state, pf_history)
    logger.log(state.hole_cards[state.actor_index])
    logger.log(bucket)
    mccfr()
    
def create_state():
    return NoLimitTexasHoldem.create_state(
        (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_DEALING,
            Automation.BOARD_DEALING,
            # Automation.HOLE_CARDS_SHOWING_OR_MUCKING, commented for now to show all hole cards at showdown
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        False,                 # ante trimming status
        0,                     # antes
        (1, 2),                # blinds
        2,                     # min bet
        (100, 100),            # starting stacks
        2,                     # player count
        mode=Mode.CASH_GAME,
    )

if __name__ == '__main__':
    logger = Logger(output_path="holdem_log.txt")
    logger.clear_logs() # clears logs so new hand can be logged

    random.seed() 
    train(100_000)
