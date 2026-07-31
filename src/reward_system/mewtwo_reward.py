"""
Team Rocket Mewtwo ex + Spidops Refactored Strategic Reward Engine.
Ground Truth: Matches 60-card CSV list (deck copy BARU CARD TAPI KECEWA.csv).
Provides bounded strategic rewards (range [-0.25, +0.25]) for RL training.
Includes TCG structural rules, item chaining bonuses, refined Giovanni gust conditions,
Psychic energy rules for Spidops, attack rewards, robust keyword opponent matching, and optional debug logging.
"""

# Card ID Constants (From Deck CSV Ground Truth)
BASIC_G_ENERGY_ID    = 1    # Basic {G} Energy
BASIC_P_ENERGY_ID    = 5    # Basic {P} Energy
TR_ENERGY_ID         = 15   # Team Rocket's Energy

TAROUNTULA_ID        = 400  # Team Rocket's Tarountula
SPIDOPS_ID           = 401  # Team Rocket's Spidops
ARTICUNO_ID          = 414  # Team Rocket's Articuno
MIMIKYU_ID           = 434  # Team Rocket's Mimikyu
MEWTWO_EX_ID         = 431  # Team Rocket's Mewtwo ex
SNEASEL_ID           = 464  # Team Rocket's Sneasel
CLEFAIRY_EX_ID       = 272  # Lillie's Clefairy ex

POFFIN_ID            = 1086 # Buddy-Buddy Poffin
SECRET_BOX_ID        = 1092 # Secret Box (ACE SPEC)
BUG_CATCHING_SET_ID  = 1094 # Bug Catching Set
ENERGY_SWITCH_ID     = 1116 # Energy Switch
ULTRA_BALL_ID        = 1121 # Ultra Ball
SACRED_ASH_ID        = 1129 # Sacred Ash
TRANSCEIVER_ID       = 1134 # Team Rocket's Transceiver
POKE_PAD_ID          = 1152 # Poké Pad
LUCKY_HELMET_ID      = 1156 # Lucky Helmet
BRAVE_BANGLE_ID      = 1175 # Brave Bangle

ARIANA_ID            = 1216 # Team Rocket's Ariana
GIOVANNI_ID          = 1218 # Team Rocket's Giovanni
PETREL_ID            = 1219 # Team Rocket's Petrel
PROTON_ID            = 1220 # Team Rocket's Proton
LILLIES_DETERM_ID    = 1227 # Lillie's Determination

TR_FACTORY_ID        = 1257 # Team Rocket's Factory

TR_POKEMON_IDS = frozenset({400, 401, 414, 431, 434, 464})
EX_POKEMON_IDS = frozenset({431, 272})
SUPPORTER_IDS  = frozenset({1216, 1218, 1219, 1220, 1227, 1134})

ITEM_IDS = frozenset({
    POFFIN_ID, SECRET_BOX_ID, BUG_CATCHING_SET_ID, ENERGY_SWITCH_ID,
    ULTRA_BALL_ID, SACRED_ASH_ID, POKE_PAD_ID, LUCKY_HELMET_ID, BRAVE_BANGLE_ID
})

# Cards strategically held in hand and EXEMPT from turn-end unplayed card penalties
STRATEGIC_HOLD_CARDS = frozenset({
    1216, 1218, 1219, 1220, 1227, 1134,  # Supporters (Only 1 Supporter per turn rule)
    1156, 1175,                          # Tools (Held for key attackers/defenders)
    1116, 1092, 1121, 1086               # Combo / Tutor items
})

# Flexible Opponent Keyword Matchers
DRAGAPULT_KEYWORDS  = ["dragapult", "pult", "dreepy", "drakloak"]
GRIMMSNARL_KEYWORDS = ["grimmsnarl", "grim", "impidimp", "morgrem"]
CHARIZARD_KEYWORDS  = ["charizard", "char", "zard", "charmander"]
MIRAIDON_KEYWORDS   = ["miraidon", "rai", "hands", "iron hands"]
LUCARIO_KEYWORDS    = ["lucario", "riolu"]
FIRE_KEYWORDS       = ["fire", "typhlosion", "centiskorch", "cyndaquil"]


def _match_opponent_archetype(opp_name: str, keyword_list: list[str]) -> bool:
    """Flexible case-insensitive substring keyword matcher for opponent archetypes."""
    name_lower = (opp_name or "").lower()
    return any(kw in name_lower for kw in keyword_list)


def calculate_mewtwo_strategic_reward(
    pre: dict,
    post: dict,
    action_type: int,
    step_idx: int,
    went_second: bool,
    player_idx: int,
    opponent_name: str = "",
    debug_log: list | None = None
) -> float:
    """
    Computes bounded strategic reward modifier r_strategic in range [-0.25, +0.25] for Mewtwo deck.
    Incorporates TCG structural rules, item chaining, attack execution rewards,
    Psychic energy logic for Spidops, turn 1 supporter guards, opponent tech bonuses, and optional debug logging.
    """
    r_strategic = 0.0
    turn = pre.get("turn", 1)

    def _add(val: float, reason: str):
        nonlocal r_strategic
        r_strategic += val
        if debug_log is not None:
            debug_log.append((reason, val))

    # -------------------------------------------------------------------------
    # 1. PLAY ACTIONS (action_type == 7)
    # -------------------------------------------------------------------------
    if action_type == 7:
        played_id = pre.get("played_card_id", -1)

        # --- A. Item Chain Sequence Bonus ---
        # Reward when >= 2 items are played in the same turn to build a setup chain
        if played_id in ITEM_IDS:
            items_this_turn = post.get("items_played_this_turn", pre.get("items_played_this_turn", 0) + 1)
            if items_this_turn >= 2:
                _add(0.06, "Item_Chain_Sequence_Bonus")

        # --- B. Basic Pokémon Benching & Opening Setup ---
        if played_id in (TAROUNTULA_ID, ARTICUNO_ID, MIMIKYU_ID, SNEASEL_ID):
            if post.get("bench_size", 0) < 4:
                val = 0.15
                if (_match_opponent_archetype(opponent_name, MIRAIDON_KEYWORDS) or
                    _match_opponent_archetype(opponent_name, LUCARIO_KEYWORDS) or
                    _match_opponent_archetype(opponent_name, FIRE_KEYWORDS)):
                    val += 0.05  # Opponent archetype tech bonus for defensive walls
                _add(val, f"Bench_Basic_{played_id}")

        elif played_id in EX_POKEMON_IDS:
            if turn <= 3 and pre.get("active_energy", 0) == 0 and post.get("bench_size", 0) == 0:
                _add(-0.15, "Avoid_Uncharged_EX_Exposure")
            elif post.get("bench_size", 0) < 4:
                _add(0.10, f"Bench_EX_{played_id}")

        # --- C. Supporter & Tutor Play ---
        elif played_id in SUPPORTER_IDS:
            # Rule Guard: Cannot play Supporter on Turn 1 if going first
            if not went_second and turn == 1:
                _add(-0.10, "Invalid_Turn1_Going_First_Supporter")
            else:
                if played_id == TRANSCEIVER_ID:
                    _add(0.18, "Play_Transceiver")
                elif played_id == ARIANA_ID:
                    _add(0.16, "Play_Ariana")
                elif played_id == GIOVANNI_ID:
                    # Refined Multi-Tiered Giovanni Gust Condition
                    opp_bench_count = len(pre.get("opp_bench_ids", []))
                    active_changed = (pre.get("opp_active_id") != post.get("opp_active_id"))

                    if active_changed and opp_bench_count > 0:
                        val = 0.18
                        if (_match_opponent_archetype(opponent_name, DRAGAPULT_KEYWORDS) or
                            _match_opponent_archetype(opponent_name, GRIMMSNARL_KEYWORDS) or
                            _match_opponent_archetype(opponent_name, CHARIZARD_KEYWORDS)):
                            val += 0.05
                        _add(val, "Successful_Giovanni_Gust")
                    elif active_changed:
                        _add(0.12, "Giovanni_Forced_Active_Reset")
                    elif opp_bench_count >= 1:
                        _add(0.08, "Giovanni_Targeted_Bench_Gust")
                    else:
                        _add(-0.05, "Wasted_Giovanni_Gust")
                elif played_id == LILLIES_DETERM_ID:
                    _add(0.15, "Play_Lillies_Determination")
                elif played_id == PROTON_ID:
                    _add(0.12, "Play_Proton")
                elif played_id == PETREL_ID:
                    _add(0.12, "Play_Petrel")

        # --- D. Search Items, Tools & Stadiums ---
        elif played_id == BUG_CATCHING_SET_ID:
            _add(0.15, "Play_Bug_Catching_Set")

        elif played_id == POFFIN_ID:
            _add(0.18, "Play_Poffin")

        elif played_id == ULTRA_BALL_ID:
            _add(0.14, "Play_Ultra_Ball")

        elif played_id == SECRET_BOX_ID:
            _add(0.16, "Play_Secret_Box")

        elif played_id == ENERGY_SWITCH_ID:
            _add(0.15, "Play_Energy_Switch")

        elif played_id == SACRED_ASH_ID:
            _add(0.12, "Play_Sacred_Ash")

        elif played_id == POKE_PAD_ID:
            _add(0.12, "Play_PokePad")

        elif played_id == BRAVE_BANGLE_ID:
            val = 0.16
            if (_match_opponent_archetype(opponent_name, DRAGAPULT_KEYWORDS) or
                _match_opponent_archetype(opponent_name, GRIMMSNARL_KEYWORDS) or
                _match_opponent_archetype(opponent_name, CHARIZARD_KEYWORDS)):
                val += 0.05
            _add(val, "Equip_Brave_Bangle")

        elif played_id == LUCKY_HELMET_ID:
            _add(0.16, "Equip_Lucky_Helmet")

        elif played_id == TR_FACTORY_ID:
            _add(0.15, "Play_TR_Factory")

    # -------------------------------------------------------------------------
    # 2. ENERGY ATTACHMENT ACTIONS (action_type == 8)
    # -------------------------------------------------------------------------
    elif action_type == 8:
        attached_id = pre.get("attached_card_id", -1)
        attached_target_id = pre.get("attached_target_id", -1)
        target_current_energy = pre.get("target_current_energy", 0)

        if attached_id == TR_ENERGY_ID:
            if attached_target_id in TR_POKEMON_IDS:
                _add(0.20, "Attach_TR_Energy_To_TR_Pokemon")
        elif attached_id == BASIC_G_ENERGY_ID:
            if attached_target_id in (TAROUNTULA_ID, SPIDOPS_ID, ARTICUNO_ID):
                _add(0.15, "Attach_Grass_Energy")
        elif attached_id == BASIC_P_ENERGY_ID:
            if attached_target_id in (CLEFAIRY_EX_ID, MEWTWO_EX_ID, MIMIKYU_ID):
                _add(0.15, "Attach_Psychic_Energy_To_Psychic_Pokemon")
            elif attached_target_id == SPIDOPS_ID:
                # Spidops attack (Wire Trap) takes {G}{C}. Psychic Energy satisfies the {C} cost!
                if target_current_energy >= 2:
                    _add(-0.10, "Overcharging_Spidops_With_Psychic_Energy")
                else:
                    _add(0.12, "Attach_Psychic_Energy_To_Spidops_For_Colorless_Cost")
            elif attached_target_id == TAROUNTULA_ID:
                _add(-0.10, "Wasteful_Psychic_Energy_On_Tarountula")

    # -------------------------------------------------------------------------
    # 3. EVOLUTION ACTIONS (action_type == 9)
    # -------------------------------------------------------------------------
    elif action_type == 9:
        evolved_id = post.get("evolved_card_id", -1)
        if evolved_id == SPIDOPS_ID:
            _add(0.20, "Evolve_Spidops")

    # -------------------------------------------------------------------------
    # 4. ATTACK ACTIONS (action_type == 13 in CABT engine)
    # -------------------------------------------------------------------------
    elif action_type in (10, 13):
        act_id = pre.get("active_id", -1)
        act_energy = pre.get("active_energy", 0)

        if act_id == MEWTWO_EX_ID and act_energy >= 2:
            val = 0.18
            if pre.get("has_brave_bangle", False) or pre.get("attached_tool_id") == BRAVE_BANGLE_ID:
                val += 0.05
            _add(val, "Mewtwo_ex_Attack_Execution")
        elif act_id in (SPIDOPS_ID, CLEFAIRY_EX_ID) and act_energy >= 2:
            _add(0.15, f"Attacker_{act_id}_Attack_Execution")
        elif act_energy >= 1:
            _add(0.10, "Basic_Attack_Execution")

    # -------------------------------------------------------------------------
    # 5. RETREAT & TACTICAL GUARDS
    # -------------------------------------------------------------------------
    elif action_type == 12:  # RETREAT
        pre_active = pre.get("active_id", -1)
        post_active = post.get("active_id", -1)
        post_active_energy = post.get("active_energy", 0)

        if pre_active in (ARTICUNO_ID, MIMIKYU_ID, TAROUNTULA_ID) and post_active in EX_POKEMON_IDS:
            if post_active_energy >= 2:
                _add(0.15, "Promote_Powered_EX_Attacker")
            else:
                _add(-0.20, "Expose_Uncharged_EX_Attacker")

    elif action_type == 14:  # END TURN
        hand_ids = pre.get("hand_ids", [])
        unplayed_basics = [cid for cid in hand_ids if cid in (TAROUNTULA_ID, ARTICUNO_ID, MIMIKYU_ID) and cid not in STRATEGIC_HOLD_CARDS]
        if len(unplayed_basics) >= 2 and post.get("bench_size", 0) <= 1 and turn > 1:
            _add(-0.15, "Unplayed_Basics_Held_With_Fragile_Bench")

    # -------------------------------------------------------------------------
    # 6. BENCH PROTECTION GUARD (Turn > 1)
    # -------------------------------------------------------------------------
    if post.get("bench_size", 0) == 0 and turn > 1:
        _add(-0.20, "Empty_Bench_Vulnerability_Turn_GT_1")

    # Final clamping in [-0.25, +0.25]
    clamped_r = max(-0.25, min(0.25, r_strategic))
    if debug_log is not None:
        debug_log.append(("Final_Clamped_Strategic_Reward", clamped_r))

    return clamped_r
