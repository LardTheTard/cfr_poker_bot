import random
import os
import pickle
import warnings
from datetime import datetime
from tqdm import tqdm
from pokerkit import Automation, Mode, NoLimitTexasHoldem, State
from collections import defaultdict
from logger import Logger
from bucketer import Bucketer
from multiprocessing import Pool

# analytics imports
import cProfile
import pstats

# sys.setrecursionlimit(10000)

warnings.filterwarnings(
    "ignore",
    message="There is no reason for this player to fold.",
    category=UserWarning,
)

def is_terminal(state: State) -> bool:
    return state.actor_index is None or state.street_index > 0

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

def mccfr(state: State, traverser: int, pf_history: list[str], nodes: dict, bucketer: Bucketer):
    """     
    """

    if is_terminal(state):
        payoff = payoff_p0(state)
        return payoff if traverser == 0 else -payoff

    bucket = bucketer.exact_preflop_bucket(state, pf_history)

    cur_actor       = state.actor_index
    actions = ['fold', 'check/call', 'raise'] # IMPLEMENT DIFFERENT RAISE SIZES: 'min_click', 'raise_medium', 'raise_big'

    if bucket not in nodes:
        nodes[bucket] = Node()
    node  = nodes[bucket]

    if cur_actor == traverser:
        # ── Traversing player: explore every action ──────────────────────────
        node.times_visited += 1

        utils = {}

        cant_raise = False
        for action in actions:
            next_state = pickle.loads(pickle.dumps(state))
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
                utils[action] = mccfr(next_state, traverser, next_pf_history, nodes, bucketer) 

        if cant_raise:
            actions = ['fold', 'check/call']

        strat = node.get_strategy(actions)

        node.accumulate_strategy(strat, actions)   # track average strategy

        node_util = sum(strat[action] * utils[action] for action in actions)
        
        for action in actions:
            node.regret_sum[action] += utils[action] - node_util
            # CHANGE FROM CFR+ to CFR due to problems with parallel merging biasing convergence
            # node.regret_sum[action] = max(node.regret_sum[action] + utils[action] - node_util, 0)

        # debug_logger.log(bucket)
        # debug_logger.log(f'utils: {utils}')
        # debug_logger.log(f'times_visited: {node.times_visited}')
        # debug_logger.log(f"regretsum: {node.regret_sum}")
        # debug_logger.log('------------------------')

        return node_util

    else:
        # ── Opponent: SAMPLE a single action ─────────────────────────────────
        next_state = pickle.loads(pickle.dumps(state))
        next_pf_history = pf_history.copy()   # ← snapshot all streets
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

        return mccfr(next_state, traverser, next_pf_history, nodes, bucketer)

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

# -- Multiprocessing / Worker Managers -------------------------------------------------------

def run_chunk(args):
    chunk_size, seed, master_nodes = args
    random.seed(seed)
    local_nodes = master_nodes.copy()
    initial_nodes = master_nodes      # already a deserialized copy — safe to keep
    local_bucketer = Bucketer()
    for count in range(chunk_size):
        state = create_state()
        play_hand(state, traverser=count % 2, nodes=local_nodes, bucketer=local_bucketer)
    return local_nodes, initial_nodes  # return both

def merge_nodes(master: dict, local: dict, initial: dict):
    for key, local_node in local.items():
        if key not in master:
            master[key] = Node()
        m = master[key]
        init = initial.get(key)
        for action, value in local_node.regret_sum.items():
            m.regret_sum[action] += value - (init.regret_sum[action] if init else 0)
        for action, value in local_node.strategy_sum.items():
            m.strategy_sum[action] += value - (init.strategy_sum[action] if init else 0)
        m.times_visited += local_node.times_visited - (init.times_visited if init else 0)

# ── Training loop ──────────────────────────────────────────────────────────────

def train(iters=100_000, n_workers=None, merge_every=100):
    if n_workers is None:
        n_workers = os.cpu_count()
    
    nodes = {}
    total_chunks = iters // merge_every  # total number of individual worker tasks
    print(f"Using {n_workers} workers, chunk size {merge_every} iterations each")

    with Pool(n_workers) as pool:
        args = [(merge_every, random.randint(0, 2**32), nodes) for _ in range(total_chunks)]
        
        with tqdm(total=total_chunks, desc="Chunks", unit="chunk") as pbar:
            for local_nodes, initial_nodes in pool.imap_unordered(run_chunk, args):
                merge_nodes(nodes, local_nodes, initial_nodes)
                pbar.update(1)
                pbar.set_postfix(nodes=len(nodes))  # shows how many info-sets discovered

                with open(f'nodesets/fullgame_1m.pkl', 'wb') as f:
                    pickle.dump(nodes, f)

    print(f"\nTraining complete ({iters:,} iterations)")
    return nodes

def play_hand(state, traverser, nodes, bucketer):
    pf_history = list()
    return mccfr(state, traverser, pf_history, nodes, bucketer)
    
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

    # cProfile.run('train(100)', 'profile_output')

    # stats = pstats.Stats('profile_output')
    # stats.sort_stats('cumulative')
    # stats.print_stats(20)  # top 20 slowest functions
    nodes = train(200_000, merge_every=100)

    for key, value in nodes.items():
        node_sum = sum(value.strategy_sum.values())
        if node_sum == 0:
            for k, v in value.strategy_sum.items():
                v = 1.0 / len(value.strategy_sum.items())

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