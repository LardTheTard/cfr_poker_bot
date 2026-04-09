import random
import os
import pickle
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
from pokerkit import Automation, Mode, NoLimitTexasHoldem, State, StandardHighHand
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
    return state.stacks[0] - state.starting_stacks[0]

# ── Info-set node ──────────────────────────────────────────────────────────────

class Node:
    def __init__(self):
        self.regret_sum   = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        self.times_visited = 0

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

    if is_terminal(state):
        payoff = payoff_p0(state)
        return payoff if traverser == 0 else -payoff

    bucket = bucketer.preflop_bucket(state, pf_history)

    cur_actor       = state.actor_index
    actions = ['fold', 'check/call', 'raise'] # IMPLEMENT DIFFERENT RAISE SIZES: 'min_click', 'raise_medium', 'raise_big'

    if bucket not in nodes:
        nodes[bucket] = Node()
    node  = nodes[bucket]
    node.times_visited += 1

    if cur_actor == traverser:
        # ── Traversing player: explore every action ──────────────────────────

        utils = {}

        cant_raise = False
        for action in actions:
            next_state = deepcopy(state)
            next_pf_history = pf_history.copy()
            match action:
                case 'fold':
                    next_state.fold()
                    next_pf_history.append('fold')
                case 'check/call':
                    next_state.check_or_call()
                    next_pf_history.append('check/call')
                case 'raise':
                    amount = get_rand_raise_size(state, bucket)
                    if next_state.can_complete_bet_or_raise_to(amount):
                        next_state.complete_bet_or_raise_to(amount)
                        next_pf_history.append('raise')
                    else:
                        cant_raise = True
            if not cant_raise:
                utils[action] = mccfr(next_state, traverser, next_pf_history) 

        if cant_raise:
            actions = ['fold', 'check/call']

        strat = node.get_strategy(actions)

        node.accumulate_strategy(strat, actions)   # track average strategy

        node_util = sum(strat[action] * utils[action] for action in actions)
        
        for action in actions:
            node.regret_sum[action] = max(node.regret_sum[action] + utils[action] - node_util, 0)

        # debug_logger.log(bucket)
        # debug_logger.log(f'utils: {utils}')
        # debug_logger.log(f'times_visited: {node.times_visited}')
        # debug_logger.log(f"regretsum: {node.regret_sum}")
        # debug_logger.log('------------------------')

        return node_util

    else:
        # ── Opponent: SAMPLE a single action ─────────────────────────────────
        next_state = deepcopy(state)
        next_pf_history = pf_history.copy()
        amount = get_rand_raise_size(state, bucket)
        if not next_state.can_complete_bet_or_raise_to(amount):
            actions = ['fold', 'check/call']
        strat = node.get_strategy(actions)
        probs = [strat[action] for action in actions]
        sampled_action = random.choices(actions, weights=probs)[0]
        if actions.index(sampled_action) == 0:
            next_state.fold()
            next_pf_history.append('fold')
        elif actions.index(sampled_action) == 1: 
            next_state.check_or_call()
            next_pf_history.append('check/call')
        elif actions.index(sampled_action) >= 2:
            next_state.complete_bet_or_raise_to(amount)
            next_pf_history.append('raise')

        # debug_logger.log(bucket) 
        # debug_logger.log(f'(OPPONENT) times_visited: {node.times_visited}')
        # debug_logger.log(f"(OPPONENT) regretsum: {node.regret_sum}")
        # debug_logger.log('------------------------')

        return mccfr(next_state, traverser, next_pf_history)

# -- Helper functions -------------------------------

def get_halfp_raise_size(state: State, bucket: tuple) -> float:
    amount = max(state.bets) + state.total_pot_amount * 1/2 #Raises half pot by default
    amount = round(amount)
    if 'vs_4bet' in bucket or amount > state.stacks[state.actor_index]:
        all_in_amt = state.stacks[state.actor_index]
        min_bet = state.min_completion_betting_or_raising_to_amount
        if min_bet is None:
            min_bet = 0
        amount = all_in_amt if all_in_amt >= min_bet else None
    return amount

def get_rand_raise_size(state: State, bucket: tuple) -> float:
    amount = max(state.bets) + state.total_pot_amount * random.choice((1/3, 1/2, 2/3, 1))
    amount = round(amount)
    if 'vs_4bet' in bucket or amount > state.stacks[state.actor_index]:
        all_in_amt = state.stacks[state.actor_index]
        min_bet = state.min_completion_betting_or_raising_to_amount
        if min_bet is None:
            min_bet = 0
        amount = all_in_amt if all_in_amt >= min_bet else None
    return amount

# ── Training loop ──────────────────────────────────────────────────────────────

def train(iters=100_000):
    """Two traversals per iteration (alternate which player is traverser)."""
    for count in tqdm(range(iters)):
        v0_state = create_state()
        v0 = play_hand(v0_state, traverser=count % 2)

    print(f"\nTraining complete ({iters:,} iterations)")

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
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs('logs', exist_ok=True)
    os.makedirs('debug_logs', exist_ok=True)
    os.makedirs('nodesets', exist_ok=True)

    logger = Logger(output_path=f"logs/{timestamp}.txt")
    debug_logger = Logger(output_path=f"debug_logs/{timestamp}.txt")

    random.seed() 
    train(200_000)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'nodesets/nodes_{timestamp}.pkl', 'wb') as f:
        pickle.dump(nodes, f)

    for key, value in nodes.items():
        logger.log(key)
        node_sum = sum(value.strategy_sum.values())
        for k, v in value.strategy_sum.items():
            logger.log(f"{k}: {v/node_sum}")
        logger.log('--------------------------------')
