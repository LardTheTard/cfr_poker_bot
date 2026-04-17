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

    def clone(self):
        new = Node()
        new.regret_sum.update(self.regret_sum)
        new.strategy_sum.update(self.strategy_sum)
        new.times_visited = self.times_visited
        return new

# ── External sampling MCCFR ────────────────────────────────────────────────────

def mccfr(state: State, traverser: int, pf_history: list[str], base_nodes: dict, delta_nodes: dict, bucketer: Bucketer):
    if is_terminal(state):
        payoff = payoff_p0(state)
        return payoff if traverser == 0 else -payoff

    bucket = bucketer.exact_preflop_bucket(state, pf_history)
    cur_actor = state.actor_index
    actions = ['fold', 'check/call', 'raise']

    base_node = base_nodes.get(bucket)

    if bucket not in delta_nodes:
        delta_nodes[bucket] = Node()
    delta_node = delta_nodes[bucket]

    def current_regret(action):
        base_val = base_node.regret_sum[action] if base_node else 0.0
        delta_val = delta_node.regret_sum[action]
        return base_val + delta_val

    def get_current_strategy(actions):
        pos = sum(max(current_regret(a), 0.0) for a in actions)
        if pos > 0:
            return {a: max(current_regret(a), 0.0) / pos for a in actions}
        return {a: 1.0 / len(actions) for a in actions}

    if cur_actor == traverser:
        delta_node.times_visited += 1
        utils = {}
        legal_actions = []

        for action in actions:
            next_state = pickle.loads(pickle.dumps(state))
            next_pf_history = pf_history.copy()

            if action == 'fold':
                next_state.fold()
                next_pf_history.append('fold')
                legal_actions.append(action)
                utils[action] = mccfr(next_state, traverser, next_pf_history, base_nodes, delta_nodes, bucketer)

            elif action == 'check/call':
                next_state.check_or_call()
                next_pf_history.append('check/call')
                legal_actions.append(action)
                utils[action] = mccfr(next_state, traverser, next_pf_history, base_nodes, delta_nodes, bucketer)

            elif action == 'raise':
                amount = get_rand_raise_size(state, bucket)
                if next_state.can_complete_bet_or_raise_to(amount):
                    next_state.complete_bet_or_raise_to(amount)
                    next_pf_history.append('raise')
                    legal_actions.append(action)
                    utils[action] = mccfr(next_state, traverser, next_pf_history, base_nodes, delta_nodes, bucketer)

        actions = legal_actions
        strat = get_current_strategy(actions)

        for action in actions:
            delta_node.strategy_sum[action] += strat[action]

        node_util = sum(strat[a] * utils[a] for a in actions)

        for action in actions:
            delta_node.regret_sum[action] += utils[action] - node_util

        return node_util

    else:
        next_state = pickle.loads(pickle.dumps(state))
        next_pf_history = pf_history.copy()

        amount = get_rand_raise_size(state, bucket)
        if not next_state.can_complete_bet_or_raise_to(amount):
            actions = ['fold', 'check/call']

        strat = get_current_strategy(actions)
        sampled_action = random.choices(actions, weights=[strat[a] for a in actions])[0]

        if sampled_action == 'fold':
            next_state.fold()
            next_pf_history.append('fold')
        elif sampled_action == 'check/call':
            next_state.check_or_call()
            next_pf_history.append('check/call')
        else:
            next_state.complete_bet_or_raise_to(amount)
            next_pf_history.append('raise')

        return mccfr(next_state, traverser, next_pf_history, base_nodes, delta_nodes, bucketer)

# -- Helper functions -------------------------------

def clone_nodes(nodes: dict) -> dict:
    return {k: v.clone() for k, v in nodes.items()}

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

    base_nodes = master_nodes
    delta_nodes = {}
    local_bucketer = Bucketer()

    for count in range(chunk_size):
        state = create_state()
        play_hand(
            state,
            traverser=count % 2,
            base_nodes=base_nodes,
            delta_nodes=delta_nodes,
            bucketer=local_bucketer,
        )

    return delta_nodes

def merge_nodes(master: dict, delta: dict):
    for key, delta_node in delta.items():
        if key not in master:
            master[key] = Node()

        m = master[key]

        for action, value in delta_node.regret_sum.items():
            m.regret_sum[action] += value

        for action, value in delta_node.strategy_sum.items():
            m.strategy_sum[action] += value

        m.times_visited += delta_node.times_visited

# ── Training loop ──────────────────────────────────────────────────────────────

def train(iters=100_000, n_workers=None, merge_every=1000):
    if n_workers is None:
        n_workers = os.cpu_count()

    nodes = {}
    total_chunks = iters // merge_every
    print(f"Using {n_workers} workers, chunk size {merge_every} iterations each")

    with Pool(n_workers) as pool:
        with tqdm(total=total_chunks, desc="Chunks", unit="chunk") as pbar:
            chunks_done = 0

            while chunks_done < total_chunks:
                batch = min(n_workers, total_chunks - chunks_done)
                args = [
                    (merge_every, random.randint(0, 2**32), nodes)
                    for _ in range(batch)
                ]

                results = pool.map(run_chunk, args)

                for delta_nodes in results:
                    merge_nodes(nodes, delta_nodes)
                    pbar.update(1)
                    pbar.set_postfix(nodes=len(nodes))

                chunks_done += batch

                with open('nodesets/exact_pf_1m.pkl', 'wb') as f:
                    pickle.dump(nodes, f)

    print(f"\nTraining complete ({iters:,} iterations)")
    return nodes

def play_hand(state, traverser, base_nodes, delta_nodes, bucketer):
    pf_history = []
    return mccfr(state, traverser, pf_history, base_nodes, delta_nodes, bucketer)
    
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
    nodes = train(5_000_000, merge_every=1000)

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