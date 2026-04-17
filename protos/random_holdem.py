from pokerkit import Automation, Mode, NoLimitTexasHoldem, State
import random

STREET_NAMES = {
    0: "Preflop",
    1: "Flop",
    2: "Turn",
    3: "River",
}

def log(message: str) -> None:
    print(message)
    with open("holdem_log.txt", "a") as f:
        f.write(message + "\n")

def clear_logs() -> None:
    with open("holdem_log.txt", "w") as f:
        f.write("")

def log_street_state(state: State) -> None:
    street_name = STREET_NAMES.get(state.street_index, f"Street {state.street_index}")
    log(f"\n=== {street_name} ===")
    log(f"Board: {state.board_cards}")
    log(f"Total pot: {state.total_pot_amount}")
    log(f"Stacks: {state.stacks}\n")

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
            log_street_state(state)
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
            log(f"Player {actor}: raises to {amount}")
        else:
            log(f"Player {actor}: {action_name}")


    log("\n=== FINAL ===")
    log(f"Hole cards: {state.hole_cards}")
    log(f"Final board: {state.board_cards}")
    log(f"Final stacks: {state.stacks}")

    return state

if __name__ == "__main__":
    clear_logs()
    play_random_heads_up_hand()