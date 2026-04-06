import random
from tqdm import tqdm
from copy import deepcopy
from pokerkit import Automation, Mode, NoLimitTexasHoldem, Pot, RunoutCountSelection, StandardHand, State, StandardHighHand
from collections import defaultdict
from logger import Logger
from bucketer import Bucketer

# sys.setrecursionlimit(10000)

bucketer = Bucketer()

def is_terminal(state: State) -> bool:
    return state.actor_index is None or state.street_index > 0#Stop simulating after pre-flop

def payoff_p0(state: State):
    while state.actor_index is not None: #Stop simulating after pre-flop so just check thru
        state.check_or_call()

    logger.log('')
    logger.log('terminal')
    logger.log(state.board_cards)

    if state.folded_status:
        return state.stacks[0] - state.starting_stacks[0]
    p0_holes = ''.join(repr(c) for c in state.hole_cards[0] if c is not None)
    p1_holes = ''.join(repr(c) for c in state.hole_cards[1] if c is not None)
    board = ''
    for i in range(5): 
        board += repr(state.board_cards[i][0])
    p0 = StandardHighHand.from_game(p0_holes, board)
    p1 = StandardHighHand.from_game(p1_holes, board)
    if p0 == p1:
        return 0
    return state.total_pot_amount / 2 if p0 > p1 else -state.total_pot_amount / 2

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

def mccfr(state: State, traverser: int, pf_history: list[str]):
    """     
    """

    logger.log('')
    logger.log(pf_history)
    logger.log(state.bets)
    logger.log(state.board_cards)

    if is_terminal(state):
        payoff = payoff_p0(state)
        return payoff if traverser == 0 else -payoff

    bucket = bucketer.preflop_bucket(state, pf_history)
    logger.log(bucket)

    cur_actor       = state.actor_index
    actions = ['fold', 'check/call', 'raise'] # IMPLEMENT DIFFERENT RAISE SIZES: 'min_click', 'raise_medium', 'raise_big'

    if bucket not in nodes:
        nodes[bucket] = Node()
    node  = nodes[bucket]
    strat = node.get_strategy(actions)

    if cur_actor == traverser:
        # ── Traversing player: explore every action ──────────────────────────
        node.accumulate_strategy(strat, actions)   # track average strategy

        utils = {}

        for action in actions:
            next_state = deepcopy(state)
            next_pf_history = pf_history.copy()
            cant_raise = False
            match action:
                case 'fold':
                    next_state.fold()
                    next_pf_history.append('fold')
                case 'check/call':
                    next_state.check_or_call()
                    next_pf_history.append('check/call')
                case 'raise':
                    amount = max(state.bets) + state.total_pot_amount * 1/2 #Raises half pot by default
                    if 'vs_4bet' in bucket or amount > state.stacks[state.actor_index]:
                        all_in_amt = state.stacks[state.actor_index]
                        min_bet = state.min_completion_betting_or_raising_to_amount
                        if min_bet is None:
                            min_bet = 0
                        amount = all_in_amt if all_in_amt >= min_bet else None
                    if next_state.can_complete_bet_or_raise_to(amount):
                        next_state.complete_bet_or_raise_to(amount)
                        next_pf_history.append('raise')
                    else:
                        cant_raise = True
            utils[action] = mccfr(next_state, traverser, next_pf_history) if not cant_raise else -100_000

        node_util = sum(strat[action] * utils[action] for action in actions)

        # Instantaneous counterfactual regret (no reach prob needed here)
        for action in actions:
            node.regret_sum[action] += utils[action] - node_util

        return node_util

    else:
        # ── Opponent: SAMPLE a single action ─────────────────────────────────
        probs = [strat[a] for a in actions]
        sampled_action = random.choices(actions, weights=probs)[0]
        next_state = deepcopy(state)
        next_pf_history = pf_history.copy()
        if actions.index(sampled_action) == 0:
            next_state.fold()
            next_pf_history.append('fold')
        elif actions.index(sampled_action) == 1: 
            next_state.check_or_call()
            next_pf_history.append('check/call')
        elif actions.index(sampled_action) >= 2:
            amount = max(state.bets) + state.total_pot_amount * 1/2 #Raises half pot by default
            if 'vs_4bet' in bucket or amount > state.stacks[state.actor_index]:
                all_in_amt = state.stacks[state.actor_index]
                min_bet = state.min_completion_betting_or_raising_to_amount
                if min_bet is None:
                    min_bet = 0
                amount = all_in_amt if all_in_amt >= min_bet else None
            if next_state.can_complete_bet_or_raise_to(amount):
                next_state.complete_bet_or_raise_to(amount)
                next_pf_history.append('raise')
            else:
                next_state.check_or_call()
                next_pf_history.append('check/call')
        return mccfr(next_state, traverser, next_pf_history)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(iters=100_000):
    """Two traversals per iteration (alternate which player is traverser)."""
    p0_total = 0.0
    for _ in tqdm(range(iters)):
        logger.log("BOOM")
        v0_state = create_state()
        v1_state = deepcopy(v0_state)
        v0 = play_hand(v0_state, traverser=0)
        # v1 = play_hand(v1_state, traverser=1)
        p0_total += v0

    print(f"\nTraining complete ({iters:,} iterations, {2*iters:,} traversals)")
    print(f"  Avg P0 value (from P0 traversals) : {p0_total/iters:+.5f}")
    print(f"  Nash equilibrium value            : {-1/18:+.5f}\n")

def play_hand(state, traverser):
    pf_history = list()
    return mccfr(state, traverser, pf_history)
    
def create_state() -> State:
    state = NoLimitTexasHoldem.create_state(
        (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
            Automation.HOLE_DEALING,
            Automation.BOARD_DEALING,
            Automation.RUNOUT_COUNT_SELECTION,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING, #commented for now to show all hole cards at showdown
            Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING,
            Automation.CHIPS_PULLING,
        ),
        False,                 # ante trimming status
        0,                     # antes
        (1, 2),                # blinds
        1,                     # min bet
        (100, 100),            # starting stacks
        2,                     # player count
        mode=Mode.CASH_GAME,
    )
    return state


if __name__ == '__main__':
    logger = Logger(output_path="holdem_log.txt")
    logger.clear_logs() # clears logs so new hand can be logged

    random.seed(42) 
    train(100_000)

    for key, value in nodes.items():
        logger.log(key)
        logger.log(value.strategy_sum)
        logger.log('--------------------------------')
