# """
# 100% Domain-General Pokémon TCG Strategic Reward Engine (Pure AlphaZero State-Transition Architecture).

# 100% Deck-Agnostic, Card-Agnostic, & ZERO Hardcoded ID Architecture:
# - Zero hardcoded card IDs, fallback ID lists, energy IDs, or name matching heuristics.
# - Direct CardType Enum (SUPPORTER=3, STADIUM=4, SPECIAL_ENERGY=6) and Skill Effect Text parsing from cg.api.
# - Typed energy attack legality validation (can_use_attack with typed energy matching).
# - 6-Feature Continuous Board Quality Estimator V(S) in [-1.0, +1.0] (Survival, Readiness, Prize, Hand, Bench Power, Deck).
# - Pure State-Transition Deltas V(S') - V(S) for supporters, evolutions, stadiums, actions, and retreats.
# - Bounded strategic components [-0.10, +0.10], total strategic reward [-0.50, +0.50].
# - 100% backward API compatibility and 18-component dictionary keys.
# """

# from typing import Dict, Any, List, Set

# # ---------------------------------------------------------------------------
# # Dynamic Card Metadata & Effect Tag Caches (Built via cg.api)
# # ---------------------------------------------------------------------------
# _CARD_TYPE_CACHE: Dict[int, Any] = {}
# _CARD_TAG_CACHE: Dict[int, Set[str]] = {}
# _CARD_MIN_ATTACK_COST: Dict[int, int] = {}
# _CARD_MAX_DAMAGE: Dict[int, int] = {}
# _CARD_HP_CACHE: Dict[int, int] = {}
# _ENERGY_VALUE_CACHE: Dict[int, int] = {}
# _CARD_TYPED_REQS_CACHE: Dict[int, Set[int]] = {}


# def _build_metadata_cache() -> None:
#     """Builds generic card metadata and effect tag caches directly from cg.api."""
#     global _CARD_TYPE_CACHE, _CARD_TAG_CACHE
#     global _CARD_MIN_ATTACK_COST, _CARD_MAX_DAMAGE, _CARD_HP_CACHE
#     global _ENERGY_VALUE_CACHE, _CARD_TYPED_REQS_CACHE

#     try:
#         from cg.api import all_card_data, all_attack, CardType
#         atk_map = {a.attackId: a for a in all_attack()}
#         for card in all_card_data():
#             cid = card.cardId
#             _CARD_TYPE_CACHE[cid] = card.cardType

#             tags: Set[str] = set()

#             # Dynamic CardType Enum Matching directly from cg.api
#             if card.cardType == CardType.SUPPORTER:
#                 tags.add("supporter")
#             elif card.cardType == CardType.STADIUM:
#                 tags.add("stadium")
#             elif card.cardType == CardType.ITEM:
#                 tags.add("item")
#             elif card.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
#                 tags.add("energy")
#                 _ENERGY_VALUE_CACHE[cid] = 2 if card.cardType == CardType.SPECIAL_ENERGY else 1

#             if hasattr(card, "hp") and card.hp:
#                 _CARD_HP_CACHE[cid] = card.hp

#             # Dynamic Skill Effect Text Parsing directly from cg.api
#             if hasattr(card, "skills") and card.skills:
#                 for s in card.skills:
#                     stext = (getattr(s, "text", "") or "").lower()
#                     if "search" in stext:
#                         tags.add("search")
#                     if "draw" in stext:
#                         tags.add("draw")

#             _CARD_TAG_CACHE[cid] = tags

#             if card.attacks:
#                 costs = [len(atk_map[aid].energies) for aid in card.attacks if aid in atk_map]
#                 damages = [getattr(atk_map[aid], "damage", 30) for aid in card.attacks if aid in atk_map]
#                 _CARD_MIN_ATTACK_COST[cid] = min(costs, default=1)
#                 _CARD_MAX_DAMAGE[cid] = max(damages, default=30)

#                 typed_set: Set[int] = set()
#                 for aid in card.attacks:
#                     if aid in atk_map:
#                         for e in atk_map[aid].energies:
#                             if e != 0:
#                                 typed_set.add(e)
#                 _CARD_TYPED_REQS_CACHE[cid] = typed_set

#     except Exception:
#         pass


# _build_metadata_cache()


# # ---------------------------------------------------------------------------
# # Domain-General Metadata Query Helpers (Zero Fallback ID Lists)
# # ---------------------------------------------------------------------------
# def is_supporter(card_id: int) -> bool:
#     if not _CARD_TAG_CACHE:
#         _build_metadata_cache()
#     return "supporter" in _CARD_TAG_CACHE.get(card_id, set())


# def is_stadium(card_id: int) -> bool:
#     if not _CARD_TAG_CACHE:
#         _build_metadata_cache()
#     return "stadium" in _CARD_TAG_CACHE.get(card_id, set())


# def is_search_card(card_id: int) -> bool:
#     if not _CARD_TAG_CACHE:
#         _build_metadata_cache()
#     tags = _CARD_TAG_CACHE.get(card_id, set())
#     return "search" in tags or "draw" in tags


# def get_energy_card_value(card_id: int) -> int:
#     if not _ENERGY_VALUE_CACHE:
#         _build_metadata_cache()
#     return _ENERGY_VALUE_CACHE.get(card_id, 1)


# def get_min_attack_cost(card_id: int) -> int:
#     if not _CARD_MIN_ATTACK_COST:
#         _build_metadata_cache()
#     return max(1, _CARD_MIN_ATTACK_COST.get(card_id, 1))


# def get_max_attack_damage(card_id: int) -> int:
#     if not _CARD_MAX_DAMAGE:
#         _build_metadata_cache()
#     return max(10, _CARD_MAX_DAMAGE.get(card_id, 30))


# def get_max_hp(card_id: int) -> int:
#     if not _CARD_HP_CACHE:
#         _build_metadata_cache()
#     return max(30, _CARD_HP_CACHE.get(card_id, 100))


# def can_use_attack(card_id: int, current_energy_count: int, energy_types: List[int] = None) -> bool:
#     """
#     Checks total energy count and matching energy types against required attack energy types.
#     """
#     min_cost = get_min_attack_cost(card_id)
#     if current_energy_count < min_cost:
#         return False

#     required_types = _CARD_TYPED_REQS_CACHE.get(card_id, set())
#     if not required_types or not energy_types:
#         return True

#     # Ignore 0 (colorless energy requirement)
#     typed_reqs = {t for t in required_types if t != 0}
#     if not typed_reqs:
#         return True

#     available_types = set(energy_types)
#     return typed_reqs.issubset(available_types)


# # ---------------------------------------------------------------------------
# # 6-Feature Continuous Board Quality Estimator V(S) in [-1.0, +1.0]
# # ---------------------------------------------------------------------------
# def calculate_board_value(state: dict) -> float:
#     """
#     Computes generic domain-agnostic board state quality V(S) in [-1.0, +1.0].
#     Preserves continuous ordering: losing state (-1.0) < neutral state (0.0) < winning state (+1.0).
#     Features:
#       - 0.25: Active survival ratio ((2 * HP/MaxHP) - 1.0)
#       - 0.20: Active attack readiness ratio ((2 * Energy/MinReq) - 1.0)
#       - 0.15: Prize advantage ratio ((opp_prizes - my_prizes) / 6.0)
#       - 0.15: Hand resource advantage ((2 * min(1.0, hand_size / 7.0)) - 1.0)
#       - 0.15: Bench quality & powered attackers ((2 * bench_powered_ratio) - 1.0)
#       - 0.10: Deck safety margin ((2 * min(1.0, deck_size / 5.0)) - 1.0)
#     """
#     act_id = state.get("active_id", -1)
#     act_hp = state.get("active_hp", 100)
#     act_max_hp = state.get("active_max_hp", get_max_hp(act_id))
#     survival_ratio = (2.0 * min(1.0, max(0.0, act_hp / max(1, act_max_hp)))) - 1.0

#     act_en = state.get("active_energy", 0)
#     min_req = state.get("active_min_req_energy", get_min_attack_cost(act_id))
#     readiness_ratio = (2.0 * min(1.0, act_en / max(1, min_req))) - 1.0

#     my_prizes = state.get("my_prizes", 6)
#     opp_prizes = state.get("opp_prizes", 6)
#     prize_adv = max(-1.0, min(1.0, (opp_prizes - my_prizes) / 6.0))

#     hand_size = state.get("hand_size", 0)
#     hand_adv = (2.0 * min(1.0, hand_size / 7.0)) - 1.0

#     bench_size = state.get("bench_size", 0)
#     bench_energies = state.get("bench_energies", [])
#     bench_powered = sum(1 for e in bench_energies if e >= 1) if bench_energies else 0
#     bench_power = (2.0 * min(1.0, (bench_size + bench_powered) / 4.0)) - 1.0

#     deck_size = state.get("deck_size", 60)
#     deck_safety = 1.0 if deck_size > 5 else ((2.0 * (deck_size / 5.0)) - 1.0)

#     val = (0.25 * survival_ratio +
#            0.20 * readiness_ratio +
#            0.15 * prize_adv +
#            0.15 * hand_adv +
#            0.15 * bench_power +
#            0.10 * deck_safety)

#     return max(-1.0, min(1.0, val))


# # ---------------------------------------------------------------------------
# # Strategic Reward Component Evaluator
# # ---------------------------------------------------------------------------
# def calculate_base_strategic_reward_components(pre: dict, post: dict, action_type: int, step_idx: int, went_second: bool, player_idx: int, opponent_name: str) -> dict:
#     """
#     Computes domain-general strategic reward components for state transitions S -> S'.
#     Zero hardcoded card IDs, energy IDs, or fallback ID lists.
#     """
#     components = {
#         "r_knockout": 0.0,
#         "r_attack_ready": 0.0,
#         "r_backup_ready": 0.0,
#         "r_bench_setup": 0.0,
#         "r_evolution_progress": 0.0,
#         "r_stadium_value": 0.0,
#         "r_retreat_eff": 0.0,
#         "r_damage_eff": 0.0,
#         "r_lethal_detection": 0.0,
#         "r_supporter_eff": 0.0,
#         "r_supporter_opp_cost": 0.0,
#         "r_hand_congestion": 0.0,
#         "r_deck_preservation": 0.0,
#         "r_missed_attack": 0.0,
#         "r_donk_prevention": 0.0,
#         "r_action_conv": 0.0,
#         "r_search_quality": 0.0,
#         "r_search_tempo": 0.0,
#     }

#     pre_val = calculate_board_value(pre)
#     post_val = calculate_board_value(post)
#     val_delta = post_val - pre_val

#     # 1. Knockout Execution & Damage Efficiency & Lethal Detection
#     pre_opp_pk = pre.get("opp_pokemon", 0)
#     post_opp_pk = post.get("opp_pokemon", 0)
#     if pre_opp_pk > post_opp_pk and post_opp_pk >= 0 and pre_opp_pk > 0:
#         components["r_knockout"] = min(0.10, (pre_opp_pk - post_opp_pk) * 0.10)

#     pre_opp_hp = pre.get("opp_active_hp", 0)
#     post_opp_hp = post.get("opp_active_hp", 0)
#     pre_opp_max_hp = pre.get("opp_active_max_hp", get_max_hp(pre.get("opp_active_id", -1)))
#     if pre_opp_hp > post_opp_hp and pre_opp_hp > 0:
#         opp_hp_loss = pre_opp_hp - post_opp_hp
#         damage_ratio = opp_hp_loss / max(1, pre_opp_max_hp)
#         components["r_damage_eff"] = min(0.10, damage_ratio * 0.10)
#         if post_opp_hp <= 0:
#             components["r_lethal_detection"] = 0.10

#     # 2. Dynamic Attack Readiness & Backup Power
#     act_id = pre.get("active_id", -1)
#     min_req_en = pre.get("active_min_req_energy", get_min_attack_cost(act_id))
#     pre_act_en = pre.get("active_energy", 0)
#     post_act_en = post.get("active_energy", 0)
#     energy_types = pre.get("active_energy_types", None)

#     pre_can_attack = can_use_attack(act_id, pre_act_en, energy_types)
#     post_can_attack = can_use_attack(act_id, post_act_en, energy_types)

#     if post_can_attack and not pre_can_attack:
#         components["r_attack_ready"] = 0.08

#     pre_bench_size = pre.get("bench_size", 0)
#     post_bench_size = post.get("bench_size", 0)
#     pre_total_en = pre.get("energy", 0)
#     post_total_en = post.get("energy", 0)
#     bench_en = post_total_en - post_act_en
#     if post_bench_size > 0 and bench_en >= min_req_en and post_total_en > pre_total_en:
#         components["r_backup_ready"] = 0.06

#     # 3. Early Bench Setup & Donk Guard
#     turn_curr = post.get("turn", 1)
#     if post_bench_size > pre_bench_size and turn_curr <= 3:
#         components["r_bench_setup"] = min(0.08, 0.04 * (post_bench_size - pre_bench_size))

#     if post_bench_size == 0 and turn_curr <= 2:
#         components["r_donk_prevention"] = -0.10

#     # 4. Evolution Progress (Pure State-Transition Delta)
#     if action_type == 9:  # EVOLVE
#         components["r_evolution_progress"] = max(-0.05, min(0.08, val_delta * 0.25))

#     # 5. Supporter Play & Opportunity Cost (Pure State-Transition Delta)
#     played_card_id = pre.get("played_card_id", -1)
#     if action_type == 7 and is_supporter(played_card_id):
#         components["r_supporter_eff"] = max(-0.08, min(0.08, val_delta * 0.30))

#     post_hand_ids = post.get("hand_ids", [])
#     has_supporter_in_hand = any(is_supporter(cid) for cid in post_hand_ids)
#     if action_type in (0, 14) and (not is_supporter(played_card_id)) and has_supporter_in_hand:  # END_TURN
#         components["r_supporter_opp_cost"] = -0.06

#     # 6. Search Quality & Tempo
#     if (action_type == 7 and is_search_card(played_card_id)) or action_type == 3:
#         components["r_search_quality"] = 0.06
#         components["r_search_tempo"] = 0.05

#     # 7. Action Conversion & Hand Congestion (Pure State-Transition Delta)
#     if action_type in (7, 8) and val_delta > 0:  # PLAY or ATTACH improving board state
#         components["r_action_conv"] = min(0.05, val_delta * 0.20)

#     post_hand_size = post.get("hand_size", 0)
#     if post_hand_size > 10 and action_type == 7:
#         components["r_hand_congestion"] = -0.05

#     # 8. Pure AlphaZero State-Transition Retreat Delta V(S -> S')
#     if action_type == 12:  # RETREAT
#         pre_min_req = pre.get("active_min_req_energy", get_min_attack_cost(pre.get("active_id", -1)))
#         pre_readiness = min(1.0, pre_act_en / max(1, pre_min_req))

#         post_act_id = post.get("active_id", -1)
#         post_min_req = post.get("active_min_req_energy", get_min_attack_cost(post_act_id))
#         post_readiness = min(1.0, post_act_en / max(1, post_min_req))

#         pre_survival = pre.get("active_hp", 100) / max(1, pre.get("active_max_hp", 100))
#         post_survival = post.get("active_hp", 100) / max(1, post.get("active_max_hp", 100))

#         readiness_delta = post_readiness - pre_readiness
#         survival_delta = post_survival - pre_survival

#         board_delta = 0.60 * readiness_delta + 0.40 * survival_delta
#         components["r_retreat_eff"] = max(-0.10, min(0.10, board_delta))

#     # 9. Deck-Out Danger Warning
#     deck_cnt_post = post.get("deck_size", 60)
#     if deck_cnt_post <= 5:
#         components["r_deck_preservation"] = -0.05 * ((6.0 - deck_cnt_post) / 5.0)

#     # 10. Stadium Deployment (Pure State-Transition Delta)
#     if action_type == 7 and is_stadium(played_card_id):
#         components["r_stadium_value"] = max(-0.08, min(0.08, val_delta * 0.30))

#     # 11. Missed Lethal Attack Penalty (Guaranteed Knockout Ignored)
#     can_KO = pre_can_attack and (pre_opp_hp <= get_max_attack_damage(act_id))
#     if action_type in (0, 14) and can_KO and (pre_opp_hp - post_opp_hp <= 0):  # END_TURN
#         components["r_missed_attack"] = -0.10

#     # Strict component bounding in [-0.10, +0.10]
#     for k in components:
#         components[k] = max(-0.10, min(0.10, components[k]))

#     return components


# def calculate_base_strategic_reward(pre: dict, post: dict, action_type: int, step_idx: int, went_second: bool, player_idx: int, opponent_name: str) -> float:
#     """Backward-compatible total scalar reward wrapper bounded in [-0.50, +0.50]."""
#     comp = calculate_base_strategic_reward_components(pre, post, action_type, step_idx, went_second, player_idx, opponent_name)
#     return max(-0.50, min(0.50, sum(comp.values())))


# def calculate_base_energy_reward(
#     action_type: int,
#     attached_card_id: int,
#     target_card_id: int,
#     target_current_energy: int,
#     bench_ids: list,
#     bench_energies: list,
#     active_id: int,
#     active_energy: int,
#     active_hp: int,
#     hand_ids: list,
#     my_prizes: int,
#     opp_prizes: int,
# ) -> float:
#     """Generic energy reward wrapper delegating to energy_evaluator (Zero hardcoded energy IDs)."""
#     if action_type != 8 or attached_card_id < 0 or target_card_id < 0:
#         return 0.0
#     try:
#         try:
#             from src.expert_system.energy_evaluator import compute_generic_energy_reward
#         except ImportError:
#             from expert_system.energy_evaluator import compute_generic_energy_reward

#         is_active = (target_card_id == active_id)
#         energy_added = get_energy_card_value(attached_card_id)
#         energy_after = target_current_energy + energy_added

#         bench_energies_after = list(bench_energies)
#         if not is_active:
#             for i, b_id in enumerate(bench_ids):
#                 if b_id == target_card_id:
#                     bench_energies_after[i] = bench_energies[i] + energy_added
#                     break

#         return compute_generic_energy_reward(
#             target_card_id=target_card_id,
#             energy_before=target_current_energy,
#             energy_after=energy_after,
#             attached_card_id=attached_card_id,
#             is_active_target=is_active,
#             bench_ids=bench_ids,
#             bench_energies_before=bench_energies,
#             bench_energies_after=bench_energies_after,
#             active_id=active_id,
#             active_energy_before=active_energy,
#         )
#     except Exception:
#         return 0.0
