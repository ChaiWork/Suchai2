"""
Team Rocket Mewtwo ex Strategic Reward Engine.
Ground Truth: Matches 60-card CSV list (deck lama.csv / deck_mewtwo.csv).
Provides bounded strategic rewards (range [-0.25, +0.25]) for RL training.
"""
from __future__ import annotations
from typing import Iterable, List, Dict, Optional

# Card ID Constants (Ground Truth: deck lama.csv)
BASIC_G_ENERGY_ID    = 1    # Basic {G} Energy (8 copies)
TR_ENERGY_ID         = 15   # Team Rocket's Energy (4 copies)

TAROUNTULA_ID        = 400  # Team Rocket's Tarountula (4 copies)
SPIDOPS_ID           = 401  # Team Rocket's Spidops (4 copies)
ARTICUNO_ID          = 414  # Team Rocket's Articuno (2 copies)
MEWTWO_EX_ID         = 431  # Team Rocket's Mewtwo ex (2 copies)
MIMIKYU_ID           = 434  # Team Rocket's Mimikyu (2 copies)

POFFIN_ID            = 1086 # Buddy-Buddy Poffin (1 copy)
BUG_CATCHING_SET_ID  = 1094 # Bug Catching Set (3 copies)
ENERGY_SWITCH_ID      = 1095 # Energy Switch (re-attach energy)
SWITCH_ID            = 1097 # Switch (2 copies)
EARTHEN_VESSEL_ID     = 1106 # Earthen Vessel (search basics, discard hand)
ULTRA_BALL_ID        = 1121 # Ultra Ball (1 copy)
TRANSCEIVER_ID       = 1134 # Team Rocket's Transceiver (4 copies)
POKE_PAD_ID          = 1152 # Poké Pad (4 copies)
MAXIMUM_BELT_ID      = 1158 # Maximum Belt (ACE SPEC Tool)
NIGHT_STRETCHER_ID   = 1159 # Night Stretcher (1 copy)
BRAVE_BANGLE_ID      = 1175 # Brave Bangle (2 copies)
PRISM_TOWER_ID        = 1180 # Prism Tower (Discard 2 -> Draw 1 Stadium)

ARIANA_ID            = 1216 # Team Rocket's Ariana (4 copies)
GIOVANNI_ID          = 1218 # Team Rocket's Giovanni (2 copies)
PETREL_ID            = 1219 # Team Rocket's Petrel (1 copy)
PROTON_ID            = 1220 # Team Rocket's Proton (2 copies)
LILLIES_DETERM_ID    = 1227 # Lillie's Determination (4 copies)

TR_FACTORY_ID        = 1257 # Team Rocket's Factory
BATTLE_CAGE_ID       = 1264 # Battle Cage (Anti-Dragapult Bench Protection) (3 copies)

TR_POKEMON_IDS = frozenset({400, 401, 414, 431, 434})
EX_POKEMON_IDS = frozenset({431})
SUPPORTER_IDS  = frozenset({1216, 1218, 1219, 1220, 1227})

ITEM_IDS = frozenset({
    POFFIN_ID, BUG_CATCHING_SET_ID, SWITCH_ID, ULTRA_BALL_ID,
    TRANSCEIVER_ID, POKE_PAD_ID, NIGHT_STRETCHER_ID, BRAVE_BANGLE_ID,
    MAXIMUM_BELT_ID, ENERGY_SWITCH_ID, EARTHEN_VESSEL_ID
})

# Standard max HP map for deck lama / Mewtwo deck Pokémon
POKEMON_MAX_HP_MAP = {
    400: 50,   # Tarountula
    401: 90,   # Spidops
    414: 120,  # Articuno
    431: 280,  # Mewtwo ex
    434: 60,   # Mimikyu
}

# Cards strategically held in hand and EXEMPT from turn-end unplayed card penalties
STRATEGIC_HOLD_CARDS = frozenset({
    1216, 1218, 1219, 1220, 1227, 1134,  # Supporters & Transceiver
    1175, 1158,                          # Tools (Brave Bangle, Maximum Belt)
    1086, 1121, 1159, 1097, 1095, 1106   # Combo / Recovery / Switch / Energy Switch / Earthen Vessel items
})

# Flexible Opponent Keyword Matchers
DRAGAPULT_KEYWORDS    = ["dragapult", "pult", "dreepy", "drakloak"]
GRIMMSNARL_KEYWORDS   = ["grimmsnarl", "grim", "impidimp", "morgrem"]
CHARIZARD_KEYWORDS    = ["charizard", "char", "zard", "charmander"]
MIRAIDON_KEYWORDS     = ["miraidon", "rai", "hands", "iron hands"]
BENCH_SNIPER_KEYWORDS = [
    "dragapult", "pult", "dreepy", "drakloak",
    "froslass", "dusknoir", "dusclops", "munkidori"
]

# =============================================================================
# STRATEGIC REWARD & PENALTY TUNING CONSTANTS (Bounded range [-0.25, +0.25])
# =============================================================================
MIN_STRATEGIC_REWARD                     = -0.25
MAX_STRATEGIC_REWARD                     = 0.25

# Penalties
PENALTY_T1_GOING_FIRST_SUPPORTER        = -0.20
PENALTY_T1_EXPOSED_MEWTWO_EX            = -0.25
PENALTY_MEWTWO_EX_ATTACK_UNDER_POWER     = -0.20
PENALTY_END_TURN_MISSED_EVOLUTION        = -0.18
PENALTY_END_TURN_MISSED_ENERGY           = -0.15
PENALTY_UNPLAYED_BATTLE_CAGE_VS_SNIPER  = -0.12
PENALTY_HIGH_BENCH_RISK_VS_SPREAD       = -0.12
PENALTY_PROMOTE_MEWTWO_EX_NO_ATTACK_NEXT = -0.10
PENALTY_EMPTY_BENCH_TURN_GT_1            = -0.08
PENALTY_BENCH_LOW_HP_BASIC_VS_SPREAD    = -0.04
PENALTY_UNPLAYED_ITEM_PER_CARD          = -0.02
MAX_UNPLAYED_ITEM_PENALTY                = -0.08

# Play Card & Sequence Rewards
REWARD_PLAY_TRANSCEIVER                  = 0.12
REWARD_PLAY_POFFIN_EARLY                 = 0.12
REWARD_REACH_FOUR_ROCKET_POKEMON         = 0.10
REWARD_PLAY_BUG_CATCHING_SET             = 0.10
REWARD_PLAY_STADIUM_TECH                 = 0.10
REWARD_EQUIP_BRAVE_BANGLE                = 0.10
REWARD_EQUIP_MAXIMUM_BELT_SNIPER         = 0.08
REWARD_PLAY_BATTLE_CAGE_EARLY_VS_SPREAD  = 0.08
REWARD_PLAY_BATTLE_CAGE_LATE_VS_SPREAD   = 0.04
REWARD_BRAVE_BANGLE_VS_META_EX           = 0.06
REWARD_PLAY_ULTRA_BALL                   = 0.08
REWARD_PLAY_POKE_PAD                     = 0.08
REWARD_PLAY_NIGHT_STRETCHER              = 0.08
REWARD_PLAY_DRAW_SUPPORTER               = 0.08
REWARD_GIOVANNI_BENCH_GUST               = 0.08
REWARD_GIOVANNI_GUST_TARGETING_EX        = 0.10
REWARD_ROCKET_SETUP_SEQUENCE             = 0.06
REWARD_ITEM_CHAIN_SEQUENCE               = 0.06
REWARD_PLAY_POFFIN_LATE                  = 0.04

# Energy Attachment Rewards
REWARD_ATTACH_ENERGY_MEWTWO_EX           = 0.14
REWARD_ATTACH_ENERGY_SPIDOPS_LINE        = 0.10
REWARD_ATTACH_ENERGY_SECONDARY_ATTACKER  = 0.08
REWARD_ATTACH_ENERGY_GENERAL             = 0.04

# Evolution Rewards
REWARD_EVOLVE_SPIDOPS                    = 0.15
REWARD_EARLY_SPIDOPS_EVOLVE_VS_AGGRO     = 0.05

# Attack Execution Rewards
REWARD_ATTACK_MEWTWO_EX                  = 0.18
REWARD_ATTACK_MEWTWO_EX_BRAVE_BANGLE     = 0.07
REWARD_MAXIMUM_BELT_KO_BONUS             = 0.05
REWARD_ATTACK_SPIDOPS                    = 0.12
REWARD_ATTACK_ARTICUNO                   = 0.10


def _match_opponent_archetype(opp_name: str, keyword_list: list[str]) -> bool:
    """Flexible case-insensitive substring keyword matcher for opponent archetypes."""
    name_lower = (opp_name or "").lower()
    return any(kw in name_lower for kw in keyword_list)


def _count_tr_pokemon(bench_ids: Iterable[int], active_id: int) -> int:
    """Counts Team Rocket Pokémon currently in play (Active + Bench)."""
    count = 0
    for cid in [active_id] + list(bench_ids):
        if cid in TR_POKEMON_IDS:
            count += 1
    return count


def _has_valid_evolution(hand_ids: Iterable[int], bench_ids: Iterable[int]) -> bool:
    """Checks if player has Spidops in hand and Tarountula on bench."""
    has_spidops_in_hand = SPIDOPS_ID in hand_ids
    has_tarountula_on_bench = any(cid == TAROUNTULA_ID for cid in bench_ids)
    return has_spidops_in_hand and has_tarountula_on_bench


def _has_valid_energy_attachment(
    hand_ids: Iterable[int],
    bench_ids: Iterable[int],
    active_id: int
) -> bool:
    """Checks if basic/special energy is present in hand and valid Pokémon targets exist."""
    has_energy = BASIC_G_ENERGY_ID in hand_ids or TR_ENERGY_ID in hand_ids
    if not has_energy:
        return False
    all_pokemon = [active_id] + list(bench_ids)
    return len(all_pokemon) > 0 and any(p > 0 for p in all_pokemon)


def _can_mewtwo_attack_next_turn(pre: dict) -> bool:
    """
    Precision check for whether Team Rocket Mewtwo ex can be readied to attack next turn.
    Requires:
    1. Power Saver condition: >= 4 Team Rocket Pokémon in play.
    2. Energy condition: >= 2 energy (or 1 energy + hand energy / special energy).
    """
    bench_ids = pre.get("bench_ids", [])
    active_id = pre.get("active_id", -1)

    # Power Saver ability check (must have >= 4 TR Pokémon in play)
    if _count_tr_pokemon(bench_ids, active_id) < 4:
        return False

    hand_ids = pre.get("hand_ids", [])
    has_g_energy = BASIC_G_ENERGY_ID in hand_ids
    has_tr_energy = TR_ENERGY_ID in hand_ids  # Special Energy count = 2

    # Check Active Mewtwo ex
    if active_id == MEWTWO_EX_ID:
        curr_e = pre.get("active_energies", pre.get("active_energy", 0))
        if curr_e >= 2 or (curr_e == 1 and (has_g_energy or has_tr_energy)) or (curr_e == 0 and has_tr_energy):
            return True

    # Check Bench Mewtwo ex
    bench_energies = pre.get("bench_energies", [])
    for idx, cid in enumerate(bench_ids):
        if cid == MEWTWO_EX_ID:
            curr_e = bench_energies[idx] if idx < len(bench_energies) else 0
            if curr_e >= 2 or (curr_e == 1 and (has_g_energy or has_tr_energy)) or (curr_e == 0 and has_tr_energy):
                return True

    return False


def _bench_risk_score_vs_spread(pre: dict, opponent_is_spread_archetype: bool) -> float:
    """
    Precisely computes bench vulnerability risk score against spread damage (e.g. Dragapult ex Phantom Dive 60 spread).
    Uses exact state fields `pre['bench_ids']` and `pre['bench_damage']`.
    Weights 2-prize EX Pokémon higher.
    """
    if not opponent_is_spread_archetype:
        return 0.0

    bench_ids = pre.get("bench_ids", [])
    bench_damage = pre.get("bench_damage", [])
    risk = 0.0

    for idx, cid in enumerate(bench_ids):
        max_hp = POKEMON_MAX_HP_MAP.get(cid, 70)
        dmg = bench_damage[idx] if idx < len(bench_damage) else 0
        remaining_hp = max_hp - dmg

        if remaining_hp <= 60:
            ko_prob = 1.0 if remaining_hp <= 40 else 0.6
            risk += ko_prob
            if cid in EX_POKEMON_IDS:
                risk += 0.5

    return risk


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
    Computes bounded strategic reward modifier r_strategic in range [-0.25, +0.25] for Team Rocket Mewtwo ex deck.
    """
    r_strategic = 0.0
    turn = pre.get("turn", 1)
    is_spread_archetype = _match_opponent_archetype(opponent_name, BENCH_SNIPER_KEYWORDS)

    def _add(val: float, reason: str):
        nonlocal r_strategic
        r_strategic += val
        if debug_log is not None:
            debug_log.append((reason, val))

    # -------------------------------------------------------------------------
    # 0. GLOBAL STATE CHECKS & SWITCH / PROMOTION GUARDS
    # -------------------------------------------------------------------------
    # A. Turn 1 going-first penalty for promoting 2-prize EX to Active
    if turn == 1 and not went_second:
        active_id = post.get("active_id", -1)
        pre_active_id = pre.get("active_id", -1)
        if active_id in EX_POKEMON_IDS and pre_active_id not in EX_POKEMON_IDS:
            _add(PENALTY_T1_EXPOSED_MEWTWO_EX, "T1_Going_First_Promoted_2Prize_EX_Active")

    # B. Mewtwo ex promotion guard & FIX 5: Reward promoting powered Mewtwo ex to active
    if action_type == 12 or (pre.get("active_id") != MEWTWO_EX_ID and post.get("active_id") == MEWTWO_EX_ID):
        if post.get("active_id") == MEWTWO_EX_ID and not _can_mewtwo_attack_next_turn(post):
            _add(PENALTY_PROMOTE_MEWTWO_EX_NO_ATTACK_NEXT, "Promoted_Mewtwo_ex_Without_Attack_Next_Turn")
        # Reward promoting Mewtwo ex when it is already fully powered (bench → active ready to attack)
        bench_mewtwo_energy = pre.get("bench_mewtwo_energy", 0)
        if post.get("active_id") == MEWTWO_EX_ID and bench_mewtwo_energy >= 3:
            tr_bench = _count_tr_pokemon(post.get("bench_ids", []), MEWTWO_EX_ID)
            if tr_bench >= 3:
                _add(0.18, "Reward_Promoted_Ready_Mewtwo_ex_To_Active")

    # FIX 1 (reward): Penalize aimless switching between 0-energy active and 0-energy bench
    if action_type == 12:
        pre_e = pre.get("my_active_energy", 0)
        post_e = post.get("my_active_energy", 0)
        pre_id = pre.get("active_id", -1)
        post_id = post.get("active_id", -1)
        if pre_e == 0 and post_e == 0 and pre_id != post_id:
            _add(-0.15, "Penalty_Pointless_Switch_Neither_Active_Powered")

    # C. End-of-turn missed development and bench risk penalties (action_type == 14)
    if action_type == 14:
        hand_ids = pre.get("hand_ids", [])
        bench_ids = pre.get("bench_ids", [])
        active_id = pre.get("active_id", -1)

        if _has_valid_evolution(hand_ids, bench_ids):
            _add(PENALTY_END_TURN_MISSED_EVOLUTION, "EndTurn_Missed_Valid_Evolution_Spidops_Line")

        if _has_valid_energy_attachment(hand_ids, bench_ids, active_id):
            _add(PENALTY_END_TURN_MISSED_ENERGY, "EndTurn_Missed_Valid_Energy_Attachment")

        bench_risk = _bench_risk_score_vs_spread(pre, is_spread_archetype)
        if bench_risk >= 1.8:
            _add(PENALTY_HIGH_BENCH_RISK_VS_SPREAD, "High_Bench_Risk_VS_Spread_Attacker")

        if is_spread_archetype and BATTLE_CAGE_ID in hand_ids and turn >= 3:
            _add(PENALTY_UNPLAYED_BATTLE_CAGE_VS_SNIPER, "Unplayed_Battle_Cage_VS_Spread_After_Turn_3")

    # -------------------------------------------------------------------------
    # 1. PLAY ACTIONS (action_type == 7)
    # -------------------------------------------------------------------------
    if action_type == 7:
        played_id = pre.get("played_card_id", -1)

        # A. Going First Turn 1 Supporter Guard Penalty
        if turn == 1 and not went_second and played_id in SUPPORTER_IDS:
            _add(PENALTY_T1_GOING_FIRST_SUPPORTER, "Invalid_Turn1_Going_First_Supporter")
            return max(MIN_STRATEGIC_REWARD, min(MAX_STRATEGIC_REWARD, r_strategic))

        # B. Team Rocket Pokémon Play & 4-Pokémon Power Saver Threshold Check
        if played_id in TR_POKEMON_IDS:
            post_bench = post.get("bench_ids", [])
            post_active = post.get("active_id", -1)
            if _count_tr_pokemon(post_bench, post_active) == 4:
                _add(REWARD_REACH_FOUR_ROCKET_POKEMON, "Reached_Four_TR_Pokemon_Power_Saver_Threshold")
            if pre.get("bench_size", 0) <= 1:
                _add(0.12, "Low_Bench_Recovery_Guard_Play_Pokemon")
            if played_id == TAROUNTULA_ID and is_spread_archetype:
                _add(PENALTY_BENCH_LOW_HP_BASIC_VS_SPREAD, "Bench_Low_HP_Basic_VS_Spread_No_Immediate_Evolution")

            # FIX 2 (reward): Early Mewtwo ex & Tarountula deployment bonus
            if turn <= 3:
                if played_id == MEWTWO_EX_ID:
                    _add(0.20, "Reward_Early_Mewtwo_ex_Deployment_Turn_1_to_3")
                elif played_id == TAROUNTULA_ID:
                    _add(0.16, "Reward_Early_Tarountula_Deployment_Turn_1_to_3")


        # C. Team Rocket's Transceiver & Setup Sequence Bonus
        if played_id == TRANSCEIVER_ID:
            _add(REWARD_PLAY_TRANSCEIVER, "Play_TR_Transceiver_Supporter_Tutor")
            items_played = post.get("items_played_this_turn", 0)
            if items_played >= 2:
                _add(REWARD_ROCKET_SETUP_SEQUENCE, "Rocket_Setup_Sequence_Transceiver_Plus")

        # D. Bug Catching Set / Earthen Vessel / Energy Switch
        elif played_id in (BUG_CATCHING_SET_ID, EARTHEN_VESSEL_ID, ENERGY_SWITCH_ID):
            _add(REWARD_PLAY_BUG_CATCHING_SET, "Play_Search_Or_Energy_Redeploy_Item")

        # E. Buddy-Buddy Poffin
        elif played_id == POFFIN_ID:
            if turn <= 2:
                _add(REWARD_PLAY_POFFIN_EARLY, "Play_Poffin_Early_Bench_Setup")
            else:
                _add(REWARD_PLAY_POFFIN_LATE, "Play_Poffin_Late_Bench_Fill")

        # F. Ultra Ball
        elif played_id == ULTRA_BALL_ID:
            _add(REWARD_PLAY_ULTRA_BALL, "Play_Ultra_Ball_Key_Search")

        # G. Poké Pad
        elif played_id == POKE_PAD_ID:
            _add(REWARD_PLAY_POKE_PAD, "Play_Poke_Pad_Supporter_Recycling")

        # H. Night Stretcher
        elif played_id == NIGHT_STRETCHER_ID:
            _add(REWARD_PLAY_NIGHT_STRETCHER, "Play_Night_Stretcher_Discard_Recovery")

        # I. Brave Bangle / Maximum Belt Equipment
        elif played_id in (BRAVE_BANGLE_ID, MAXIMUM_BELT_ID):
            if played_id == MAXIMUM_BELT_ID:
                _add(REWARD_EQUIP_MAXIMUM_BELT_SNIPER, "Equip_Maximum_Belt_30HP_Snipe_Tool")
            else:
                _add(REWARD_EQUIP_BRAVE_BANGLE, "Equip_Brave_Bangle_Attacker_Boost")
            if _match_opponent_archetype(opponent_name, DRAGAPULT_KEYWORDS + GRIMMSNARL_KEYWORDS):
                _add(REWARD_BRAVE_BANGLE_VS_META_EX, "Brave_Bangle_Tech_VS_Meta_EX")

        # J. Stadium Play (TR Factory / Prism Tower / Battle Cage)
        elif played_id in (TR_FACTORY_ID, PRISM_TOWER_ID, BATTLE_CAGE_ID):
            _add(REWARD_PLAY_STADIUM_TECH, "Play_Stadium_Tech")
            if played_id == BATTLE_CAGE_ID and is_spread_archetype:
                if turn <= 3:
                    _add(REWARD_PLAY_BATTLE_CAGE_EARLY_VS_SPREAD, "Battle_Cage_Early_VS_Spread_Archetype")
                else:
                    _add(REWARD_PLAY_BATTLE_CAGE_LATE_VS_SPREAD, "Battle_Cage_Late_VS_Spread_Archetype")

        # K. Team Rocket's Giovanni Bench Gust
        elif played_id == GIOVANNI_ID:
            opp_bench = pre.get("opp_bench_ids", [])
            if len(opp_bench) > 0:
                _add(REWARD_GIOVANNI_BENCH_GUST, "Giovanni_Targeted_Bench_Gust")
                opp_act = pre.get("opp_active_id", -1)
                if opp_act in EX_POKEMON_IDS or any(b in EX_POKEMON_IDS for b in opp_bench):
                    _add(REWARD_GIOVANNI_GUST_TARGETING_EX, "Giovanni_Gust_Targeting_EX_Prize_Swing")

        # L. Team Rocket's Ariana / Lillie's Determination Draw
        elif played_id in (ARIANA_ID, LILLIES_DETERM_ID, PROTON_ID, PETREL_ID):
            _add(REWARD_PLAY_DRAW_SUPPORTER, "Play_TR_Draw_Supporter")

        # M. Item Chain Sequence Bonus
        if played_id in ITEM_IDS and post.get("items_played_this_turn", 0) >= 2:
            _add(REWARD_ITEM_CHAIN_SEQUENCE, "Item_Chain_Sequence_Bonus")

    # -------------------------------------------------------------------------
    # 2. ENERGY ATTACHMENT (action_type == 8)
    # -------------------------------------------------------------------------
    elif action_type == 8:
        attached_id = pre.get("attached_card_id", -1)
        target_id = pre.get("attached_target_id", -1)
        active_id = pre.get("active_id", -1)
        active_energy = pre.get("my_active_energy", 0)
        req_energy_active = 3 if active_id == MEWTWO_EX_ID else (2 if active_id in (SPIDOPS_ID, ARTICUNO_ID) else 1)
        active_is_maxed = (active_energy >= req_energy_active)


        # Exact per-attacker energy hard caps
        ENERGY_HARD_CAP = {MEWTWO_EX_ID: 3, SPIDOPS_ID: 2, ARTICUNO_ID: 2, MIMIKYU_ID: 1, TAROUNTULA_ID: 1}
        target_energy = pre.get("target_energy", 0)  # Energy already on the target

        # FIX 2: Hard cap penalty for over-stacking any attacker beyond their max
        if target_id in ENERGY_HARD_CAP and target_energy >= ENERGY_HARD_CAP[target_id]:
            _add(-0.20, f"Penalty_Energy_HardCap_Exceeded_On_{target_id}")
            return max(MIN_STRATEGIC_REWARD, min(MAX_STRATEGIC_REWARD, r_strategic))

        # Guard: Articuno should NEVER waste Team Rocket's Energy!
        if attached_id == TR_ENERGY_ID and target_id == ARTICUNO_ID:
            _add(-0.20, "Penalty_TR_Energy_Wasted_On_Articuno")
        elif attached_id == BASIC_G_ENERGY_ID and target_id == ARTICUNO_ID and active_energy < 1:
            _add(0.10, "Reward_Basic_Grass_Energy_To_Articuno_For_1Energy_Retreat")
        elif attached_id in (BASIC_G_ENERGY_ID, TR_ENERGY_ID):
            if target_id == active_id and active_is_maxed:
                _add(-0.10, "Penalty_Over_Attaching_Energy_To_Already_Maxed_Active")
            elif target_id != active_id and active_is_maxed:
                if target_id == MEWTWO_EX_ID:
                    _add(0.14, "Reward_Bench_Energy_To_Mewtwo_ex_Maxed_Active")
                elif target_id in (SPIDOPS_ID, TAROUNTULA_ID):
                    _add(0.12, "Reward_Bench_Energy_To_Spidops_Line_Maxed_Active")
                else:
                    _add(0.08, "Reward_Bench_Energy_General_Maxed_Active")
            elif target_id == MEWTWO_EX_ID:
                _add(REWARD_ATTACH_ENERGY_MEWTWO_EX, "Attach_Energy_To_Mewtwo_ex_Main_Attacker")
            elif target_id in (SPIDOPS_ID, TAROUNTULA_ID):
                _add(REWARD_ATTACH_ENERGY_SPIDOPS_LINE, "Attach_Energy_To_Spidops_Line")
            elif target_id in (ARTICUNO_ID, MIMIKYU_ID):
                _add(REWARD_ATTACH_ENERGY_SECONDARY_ATTACKER, "Attach_Energy_To_Secondary_Attacker")
            else:
                _add(REWARD_ATTACH_ENERGY_GENERAL, "Attach_Energy_General")



    # -------------------------------------------------------------------------
    # 3. EVOLUTION (action_type == 9)
    # -------------------------------------------------------------------------
    elif action_type == 9:
        evolved_id = pre.get("evolved_card_id", -1)
        if evolved_id == SPIDOPS_ID:
            _add(REWARD_EVOLVE_SPIDOPS, "Evolve_Tarountula_To_Spidops")
            if turn <= 3:
                _add(REWARD_EARLY_SPIDOPS_EVOLVE_VS_AGGRO, "Early_Spidops_Evolution_VS_Aggro")

    # -------------------------------------------------------------------------
    # 4. ATTACK EXECUTION (action_type == 13)
    # -------------------------------------------------------------------------
    elif action_type == 13:
        act_id = pre.get("active_id", -1)
        bench_ids = pre.get("bench_ids", [])
        tr_count = _count_tr_pokemon(bench_ids, act_id)
        opp_active_hp = pre.get("opp_active_hp", 9999)
        act_energy = pre.get("my_active_energy", 0)

        if act_id == MEWTWO_EX_ID:
            if tr_count < 4:
                _add(PENALTY_MEWTWO_EX_ATTACK_UNDER_POWER, "Mewtwo_ex_Attack_With_Power_Saver_Unmet")
            else:
                _add(REWARD_ATTACK_MEWTWO_EX, "Mewtwo_ex_Attack_Execution")
                # FIX 4: Reward attacking in lethal window (opp HP <= ~160 = 1-shot range)
                if act_energy >= 3 and opp_active_hp <= 160:
                    _add(0.15, "Reward_Mewtwo_ex_Lethal_Window_Attack")
                if pre.get("has_brave_bangle", False):
                    _add(REWARD_ATTACK_MEWTWO_EX_BRAVE_BANGLE, "Mewtwo_ex_Brave_Bangle_Boosted_Attack")
                if pre.get("has_maximum_belt", False):
                    _add(REWARD_MAXIMUM_BELT_KO_BONUS, "Mewtwo_ex_Maximum_Belt_Boosted_Attack")
        elif act_id == SPIDOPS_ID:
            # Venomous Whip: 20 + 20 per benched Pokemon (max 5 bench = 120 damage)
            bench_size = len(bench_ids)
            spidops_damage = 20 + (20 * bench_size)  # Theoretical damage this turn
            opp_hp = pre.get("opp_active_hp", 9999)
            if bench_size == 5:
                _add(0.18, "Spidops_Venomous_Whip_Full_Bench_Max_Damage_120")
            elif bench_size >= 3:
                _add(0.12, f"Spidops_Venomous_Whip_Good_Bench_{bench_size}_Damage_{spidops_damage}")
            elif bench_size <= 1:
                _add(-0.10, f"Penalty_Spidops_Attack_Thin_Bench_{bench_size}_Only_{spidops_damage}_Damage")
            else:
                _add(REWARD_ATTACK_SPIDOPS, "Spidops_Retreat_Trap_Attack")
            # Bonus if Spidops can KO the opponent this turn
            if spidops_damage >= opp_hp > 0:
                _add(0.15, f"Spidops_Lethal_KO_Window_Bench{bench_size}_Dmg{spidops_damage}_OppHP{opp_hp}")
        elif act_id == ARTICUNO_ID:
            _add(REWARD_ATTACK_ARTICUNO, "Articuno_Snipe_Attack")

    # -------------------------------------------------------------------------
    # 5. PASS / TURN END PENALTIES (action_type == 14)
    # -------------------------------------------------------------------------
    elif action_type == 14:
        hand_ids = pre.get("hand_ids", [])
        bench_ids = pre.get("bench_ids", [])
        active_id_end = pre.get("active_id", -1)
        active_energy_end = pre.get("my_active_energy", 0)
        tr_count_end = _count_tr_pokemon(bench_ids, active_id_end)
        bench_size_end = pre.get("bench_size", len(bench_ids))

        if turn > 1 and post.get("bench_size", 0) == 0:
            _add(PENALTY_EMPTY_BENCH_TURN_GT_1, "Empty_Bench_Vulnerability_Turn_GT_1")

        # FIX 1: Penalize ending turn with powered Mewtwo ex active but not attacking
        if active_id_end == MEWTWO_EX_ID and active_energy_end >= 3 and tr_count_end >= 4:
            _add(-0.25, "Penalty_EndTurn_Powered_Mewtwo_ex_Did_Not_Attack")

        # FIX 3: Penalize ending turn with empty bench when basics are in hand
        BASIC_POKEMON_IDS = {MEWTWO_EX_ID, SPIDOPS_ID, TAROUNTULA_ID, ARTICUNO_ID, MIMIKYU_ID}
        has_basic_in_hand = any(cid in BASIC_POKEMON_IDS for cid in hand_ids)
        if bench_size_end == 0 and has_basic_in_hand and turn > 1:
            _add(-0.20, "Penalty_EndTurn_No_Bench_With_Playable_Basics_In_Hand")

        if is_spread_archetype and BATTLE_CAGE_ID in hand_ids:
            _add(PENALTY_UNPLAYED_BATTLE_CAGE_VS_SNIPER, "Penalty_Unplayed_Battle_Cage_VS_Bench_Sniper")

        unplayed_playable = [
            cid for cid in hand_ids
            if cid in ITEM_IDS and cid not in STRATEGIC_HOLD_CARDS
        ]
        if len(unplayed_playable) > 0:
            penalty = min(MAX_UNPLAYED_ITEM_PENALTY, len(unplayed_playable) * PENALTY_UNPLAYED_ITEM_PER_CARD)
            _add(penalty, "Unplayed_Playable_Items_Turn_End")

    return max(MIN_STRATEGIC_REWARD, min(MAX_STRATEGIC_REWARD, r_strategic))
