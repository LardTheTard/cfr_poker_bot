from pokerkit import Automation, Mode, NoLimitTexasHoldem, State
from logger import Logger
from bucketer import Bucketer
import random

def play_random_heads_up_hand() -> State:
    state = NoLimitTexasHoldem.create_state(
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

    last_street = None

    while state.actor_index is not None:
        # Log once whenever a new street begins
        if state.street_index != last_street:
            logger.log_street_state(state)
            last_street = state.street_index

        actions = []

        # if state.can_fold():
        #     actions.append(("fold", lambda: state.fold()))

        if state.can_check_or_call():
            actions.append(("check/call", lambda: state.check_or_call()))

        amount = 0
        if state.can_complete_bet_or_raise_to():
            min_to = state.min_completion_betting_or_raising_to_amount
            max_to = state.max_completion_betting_or_raising_to_amount

            def random_raise():
                nonlocal amount
                amount = random.randint(int(min_to), int(max_to))

                # ----------------------------- Hardcoded raise values -----------------------------
                amount = min_to
                # -------------------------------------------------------------------------------
                
                return state.complete_bet_or_raise_to(amount)

            actions.append(("raise", random_raise))

        if not actions:
            break

        action_name, action_fn = random.choice(actions)
        actor = state.actor_index
        action_fn()
        if action_name == 'raise':
            logger.log(f"Player {actor}: raises to {amount}")
        else:
            logger.log(f"Player {actor}: {action_name}")


    logger.log("\n=== FINAL ===")
    logger.log(f"Hole cards: {state.hole_cards}")
    logger.log(f"Final board: {state.board_cards}")
    logger.log(f"Final stacks: {state.stacks}")

    return state

def random_action(state: State) -> tuple:

    actions = []

    # ----------------------------- removed folding from random action pool -----------------------------
    # if state.can_fold():
    #     actions.append(("fold", lambda: state.fold()))
    # ----------------------------------------------------------

    if state.can_check_or_call():
        actions.append(("check/call", lambda: state.check_or_call()))

    amount = 0
    if state.can_complete_bet_or_raise_to():
        min_to = state.min_completion_betting_or_raising_to_amount
        max_to = state.max_completion_betting_or_raising_to_amount

        def random_raise():
            nonlocal amount
            amount = random.randint(int(min_to), int(max_to))

            # ----------------------------- Hardcoded raise values -----------------------------
            amount = min_to
            # -------------------------------------------------------------------------------
            return state.complete_bet_or_raise_to(amount)

        actions.append(("raise", random_raise))

    if not actions:
        raise Exception("No actions avaliable???")

    action_name, action_fn = random.choice(actions)
    action_fn()
    return action_name, amount

def play_one_sided_heads_up_hand() -> State:
    state = NoLimitTexasHoldem.create_state(
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

    last_street = None

    while state.actor_index is not None:
        # Log once whenever a new street begins
        if state.street_index != last_street:
            logger.log_street_state(state)
            last_street = state.street_index

        actor = state.actor_index
        if actor == 0:
            action_name, amount = random_action(state)
        else:
            while (True):
                try:
                    action = int(input("Action? 0: fold, 1: check/call, 2: raise\n"))
                    match action:
                        case 0:
                            state.fold()
                            action_name = "fold"
                        case 1:
                            state.check_or_call()
                            action_name = "check/call"
                        case 2:
                            amount = int(input("Input raise size: "))
                            state.complete_bet_or_raise_to(amount)
                            action_name = "raise"
                    break
                except Exception:
                    print("Error. Try reinputting your action.\n")
        if action_name == 'raise':
            logger.log(f"Player {actor}: raises to {amount}")
        else:
            logger.log(f"Player {actor}: {action_name}")

    logger.log_final(state)

    return state

def bucket_play_one_sided_heads_up_hand() -> State:
    state = NoLimitTexasHoldem.create_state(
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

    print(tuple(state.pot_amounts))
    print("boom")

    bucketer = Bucketer()
    pf_history = []
    flop_history = []
    turn_history = []
    river_history = []

    last_street = None

    while state.actor_index is not None:
        # Log once whenever a new street begins
        match state.street_index:
            case 0: # IF PREFLOP: 
                bucket = bucketer.preflop_bucket(state, pf_history)
                logger.log(state.hole_cards[state.actor_index])
                logger.log(bucket)

        if state.street_index != last_street:
            logger.log_street_state(state)
            last_street = state.street_index

        if state.actor_index == 0:
            action_name, amount = random_action(state)
        else:
            while (True):
                try:
                    action = int(input("Action? 0: fold, 1: check/call, 2: raise\n"))
                    match action:
                        case 0:
                            state.fold()
                            action_name = "fold"
                        case 1:
                            state.check_or_call()
                            action_name = "check/call"
                        case 2:
                            amount = int(input("Input raise size: "))
                            if amount >= state.stacks[state.actor_index]:
                                amount = state.stacks[state.actor_index]
                            state.complete_bet_or_raise_to(amount)
                            action_name = "raise"
                    break
                except Exception:
                    print("Error. Try reinputting your action.\n")
        if action_name == 'raise':
            logger.log(f"Player {state.actor_index}: raises to {amount}")
        else:
            logger.log(f"Player {state.actor_index}: {action_name}")

        # Per street history recording
        match state.street_index:
            case 0:
                pf_history.append(action_name)
            case 1:
                flop_history.append(action_name)
            case 2:
                turn_history.append(action_name)
            case 3:
                river_history.append(action_name)

    logger.log_final(state)

    return state

if __name__ == "__main__":
    logger = Logger(output_path="holdem_log.txt")
    logger.clear_logs() # clears logs so new hand can be logged
    bucket_play_one_sided_heads_up_hand()
    # play_one_sided_heads_up_hand()