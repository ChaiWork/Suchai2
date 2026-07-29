"""
Strategic and Step Reward Engine for Pokémon TCG Reinforcement Learning Agent.

Computes bounded step rewards [-0.15, +0.15] and strategic heuristic modifiers
(r_strategic) to guide credit assignment during GAE advantage calculation.
"""

from cg.api import CardType, OptionType, SelectContext, EnergyType
from training.card_database import (
    get_card_data, is_defensive_blocker, can_attack, attack_table
)


def calculate_strategic_reward(pre, post, action_type, step_idx, went_second, player_idx, opponent_name):
    """
    Computes r_strategic (heuristic state transition modifier).
    
    Returns:
        float: r_strategic value for this transition.
    """
    r_strategic = 0.0

    deck_cnt_post = post.get("deck_count", 60)
    if deck_cnt_post <= 6 and action_type == 7:
        r_strategic -= 0.35

    if step_idx <= 2:
        bench_size_curr = post.get("bench_size", 0)
        if bench_size_curr == 0:
            r_strategic -= 0.50
        elif bench_size_curr >= 2:
            r_strategic += 0.25

    opp_pk_lost = pre.get("opp_pokemon", 0) - post.get("opp_pokemon", 0)
    if opp_pk_lost > 0:
        pre_opp_b = pre.get("opp_bench_ids", [])
        post_opp_b = post.get("opp_bench_ids", [])
        for cid in pre_opp_b:
            if cid not in post_opp_b:
                card = get_card_data(cid)
                if card and getattr(card, "name", None) and any(p in card.name.lower() for p in ["dreepy", "snover", "drakloak", "tarountula", "abra", "staryu", "impidimp", "phantump", "applin", "riolu", "duskull", "dusclops", "duraludon", "solrock", "lunatone", "gible", "gabite", "froslass", "snorunt"]):
                    r_strategic += 0.15
                    break

    if action_type == 7:  # PLAY
        played_id = pre.get("played_card_id", -1)
        played_card = get_card_data(played_id)
        
        if played_card is not None and played_card.cardType == CardType.POKEMON and played_card.basic:
            opp_b_ids_post = post.get("opp_bench_ids", [])
            opp_act_id_post = post.get("opp_active_id", -1)
            is_dragapult = any(get_card_data(cid) is not None and any(kw in get_card_data(cid).name.lower() for kw in ["dreepy", "drakloak", "dragapult"]) for cid in opp_b_ids_post + [opp_act_id_post])
            max_bench_limit = 3 if is_dragapult else 4
            if post.get("bench_size", 0) >= max_bench_limit:
                r_strategic -= 0.15
        if played_id in (431, 272):
            r_strategic += 0.20

        if played_id in (1257, 1258) and pre.get("stadium_id") is not None and pre.get("stadium_id") in (1244, 1259, 1256):
            r_strategic += 0.25

        my_hand_size = pre.get("hand_size", 5)
        if my_hand_size <= 2 and played_id in (1257, 1134, 1097, 1094, 1216):
            r_strategic += 0.15

        if played_id in (1159, 1175):
            r_strategic += 0.15

    elif action_type == 13:  # ATTACK
        opp_act_id = pre.get("opp_active_id", -1)
        opp_act_card = get_card_data(opp_act_id)
        if opp_act_card and getattr(opp_act_card, "name", None) and any(kw in opp_act_card.name.lower() for kw in ["riolu", "lucario"]):
            r_strategic += 0.20

        if pre.get("active_hp", 999) <= 50 and opp_act_id == 345:
            r_strategic -= 0.35

    if post.get("active_id") in (414, 434) and post.get("active_energy", 0) == 0:
        opp_act_id = pre.get("opp_active_id", -1)
        if opp_act_id in (723, 722, 1145):
            r_strategic -= 0.20

    if went_second and step_idx == 0:
        if action_type == 13:
            r_strategic += 0.35
        elif action_type in [7, 8]:
            r_strategic += 0.15

    is_tr_pokemon = lambda cid: (get_card_data(cid) is not None and 
                                 get_card_data(cid).cardType == CardType.POKEMON and 
                                 "rocket" in get_card_data(cid).name.lower())

    active_id_post = post.get("active_id", -1)
    bench_ids_post = post.get("bench_ids", [])
    tr_count = 0
    if is_tr_pokemon(active_id_post):
        tr_count += 1
    for bid in bench_ids_post:
        if is_tr_pokemon(bid):
            tr_count += 1
    
    if tr_count < 4:
        if action_type == 7:
            played_id = pre.get("played_card_id", -1)
            if is_tr_pokemon(played_id):
                r_strategic += 0.25
        if action_type == 8 and tr_count < 3:
            attached_target = pre.get("attached_target_id", -1)
            if attached_target == 431:
                r_strategic -= 0.10
    
    if pre.get("context") == SelectContext.TO_ACTIVE:
        promoted_id = post.get("active_id", -1)
        promoted_energy = post.get("active_energies", 0)
        
        promoted_c = get_card_data(promoted_id)
        promoted_can_attack = can_attack(promoted_id, promoted_energy)
        promoted_is_blocker = is_defensive_blocker(promoted_c)
        
        benched_options = []
        for idx, bid in enumerate(pre.get("bench_ids", [])):
            b_c = get_card_data(bid)
            if b_c is not None and b_c.cardType == CardType.POKEMON:
                b_energy = pre.get("bench_energies", [])[idx] if idx < len(pre.get("bench_energies", [])) else 0
                b_can_attack = can_attack(bid, b_energy)
                b_is_blocker = is_defensive_blocker(b_c)
                benched_options.append((bid, b_energy, b_can_attack, b_is_blocker))
                
        if len(benched_options) > 0:
            is_meaningful = promoted_can_attack or promoted_is_blocker
            bench_has_ready_attacker = any(opt[2] for opt in benched_options)
            
            if bench_has_ready_attacker and not promoted_can_attack:
                r_strategic -= 0.50
            elif not is_meaningful:
                r_strategic -= 0.20
            elif promoted_is_blocker:
                opp_active_card = get_card_data(pre.get("opp_active_id"))
                opp_is_ex = opp_active_card is not None and (getattr(opp_active_card, "ex", False) or getattr(opp_active_card, "megaEx", False) or "ex" in getattr(opp_active_card, "name", "").lower())
                r_strategic += 0.20
            
            opp_act_c = get_card_data(pre.get("opp_active_id"))
            if promoted_id == 434 and opp_act_c is not None and getattr(opp_act_c, "energyType", None) == EnergyType.DARKNESS:
                r_strategic -= 0.50
    
    if pre.get("stadium_id") != post.get("stadium_id") and post.get("stadium_id") != -1:
        stadium_card = get_card_data(post.get("stadium_id"))
        if stadium_card is not None and stadium_card.cardType == CardType.STADIUM:
            if action_type == 7:
                r_strategic += 0.10
    
    if action_type == 7:  # PLAY
        played_id = pre.get("played_card_id", -1)
        played_card = get_card_data(played_id)
        
        if played_card is not None:
            if played_id in [1116, 1159, 1175, 1121, 1134, 1257]:
                r_strategic += 0.08
                
            if played_card.cardType == CardType.POKEMON and not getattr(played_card, "stage1", False) and not getattr(played_card, "stage2", False):
                pre_bench_count = len(pre.get("bench_ids", []))
                if pre_bench_count == 0:
                    r_strategic += 0.40
                elif played_card.cardId == 431:
                    r_strategic += 0.35
                    if pre.get("turn", 1) <= 5:
                        r_strategic += 0.25
                elif played_card.cardId in [400, 434, 414]:
                    r_strategic += 0.25
                else:
                    r_strategic += 0.15
                        
            elif played_card.cardType in [CardType.ITEM, CardType.SUPPORTER] and any(
                kw in played_card.name.lower() or kw in getattr(played_card, "text", "").lower() 
                for kw in ["search", "draw", "look", "put"]
            ):
                my_deck_size = pre.get("deck_size", 40)
                if my_deck_size <= 5:
                    r_strategic -= 0.10
                else:
                    expected_min_diff = -2 if "Ultra Ball" in played_card.name else 0
                    hand_diff = post["hand_size"] - pre["hand_size"]
                    if hand_diff >= expected_min_diff:
                        r_strategic += 0.10
                    
                    if "Ultra Ball" in played_card.name:
                        pre_hand = list(pre.get("hand_ids", []))
                        post_hand = list(post.get("hand_ids", []))
                        if played_id in pre_hand:
                            pre_hand.remove(played_id)
                            
                        from collections import Counter
                        pre_counts = Counter(pre_hand)
                        post_counts = Counter(post_hand)
                        discarded_ids = []
                        for cid, count in pre_counts.items():
                            diff = count - post_counts.get(cid, 0)
                            if diff > 0:
                                discarded_ids.extend([cid] * diff)
                                
                        efficiency_score = 0.0
                        for dcid in discarded_ids:
                            dc = get_card_data(dcid)
                            if dc is not None:
                                if dc.cardType == CardType.BASIC_ENERGY:
                                    has_spidops = any(bid == 401 for bid in pre.get("bench_ids", [])) or pre.get("active_id") == 401
                                    has_stretcher = 1097 in post_hand
                                    if has_spidops or has_stretcher:
                                        efficiency_score += 0.10
                                elif post_counts.get(dcid, 0) > 0:
                                    efficiency_score += 0.05
                                elif dc.cardType == CardType.STADIUM and pre.get("stadium_id") == dcid:
                                    efficiency_score += 0.08
                                elif dc.cardType in [CardType.SUPPORTER, CardType.TOOL]:
                                    efficiency_score -= 0.10
                        r_strategic += efficiency_score
                    else:
                        r_strategic -= 0.05
                    
            elif played_card.cardType == CardType.ITEM and "Energy Switch" in played_card.name:
                active_c = get_card_data(post["active_id"])
                if active_c is not None:
                    active_att_costs = [len(attack_table.get(aid).energies) for aid in active_c.attacks if attack_table.get(aid)]
                    min_att_cost = min(active_att_costs) if active_att_costs else 99
                    if post["active_energies"] >= min_att_cost and pre["active_energies"] < min_att_cost:
                        r_strategic += 0.25
                    elif post["active_energies"] > pre["active_energies"]:
                        r_strategic += 0.10
                    else:
                        r_strategic -= 0.05
            elif "Transceiver" in played_card.name:
                hand_diff = post["hand_size"] - pre["hand_size"]
                if hand_diff >= 0:
                    r_strategic += 0.10
                else:
                    r_strategic -= 0.05
                    
            elif played_id == 1086:
                pre_bench  = len(pre.get("bench_ids", []))
                post_bench = len(post.get("bench_ids", []))
                mons_benched = post_bench - pre_bench
                if mons_benched >= 2:
                    r_strategic += 0.30
                elif mons_benched == 1:
                    r_strategic += 0.20
                else:
                    r_strategic += 0.10

            elif played_id == 1123:
                pre_active_id  = pre.get("active_id", -1)
                post_active_id = post.get("active_id", -1)
                post_active_c  = get_card_data(post_active_id)
                is_post_ready  = post_active_c is not None and post.get("active_energies", 0) >= 1
                if post_active_id in [414, 401, 431] and is_post_ready:
                    r_strategic += 0.25
                elif pre_active_id == 431 and pre.get("active_hp", 280) <= 140:
                    r_strategic += 0.20
                else:
                    r_strategic += 0.10

            elif played_id == 1094:
                post_hand = post.get("hand_ids", [])
                pre_hand  = pre.get("hand_ids", [])
                grass_mon_ids = [400, 401]
                grass_energy_id = 1
                found_grass_mon    = any(cid in post_hand and cid not in pre_hand for cid in grass_mon_ids)
                found_grass_energy = post_hand.count(grass_energy_id) > pre_hand.count(grass_energy_id)
                spidops_in_play = (pre.get("active_id") == 401 or 401 in pre.get("bench_ids", []))
                if found_grass_mon and not spidops_in_play:
                    r_strategic += 0.25
                elif found_grass_mon:
                    r_strategic += 0.15
                elif found_grass_energy:
                    r_strategic += 0.12
                else:
                    r_strategic -= 0.08

            elif played_id == 1097:
                pre_discard  = pre.get("discard_ids", [])
                post_hand    = post.get("hand_ids", [])
                pre_hand     = pre.get("hand_ids", [])
                attacker_ids = [431, 401, 400]
                energy_ids   = [15, 1, 5]
                got_attacker = any(cid in post_hand and cid not in pre_hand for cid in attacker_ids)
                got_energy   = any(cid in post_hand and cid not in pre_hand for cid in energy_ids)
                has_attacker_discard = any(cid in pre_discard for cid in attacker_ids)
                prizes_opp = pre.get("prizes_remaining_opp", 3)
                if got_attacker and prizes_opp <= 2:
                    r_strategic += 0.40
                elif got_attacker:
                    r_strategic += 0.25
                elif got_energy:
                    r_strategic += 0.15
                elif has_attacker_discard and not got_attacker:
                    r_strategic -= 0.10
                else:
                    r_strategic += 0.05

            elif played_id == 1217:
                post_bench_energies = post.get("bench_energies", [])
                pre_bench_energies  = pre.get("bench_energies", [])
                bench_energy_gained = sum(post_bench_energies) - sum(pre_bench_energies)
                active_energy_gained = post.get("active_energies", 0) - pre.get("active_energies", 0)
                total_energy_gained = bench_energy_gained + active_energy_gained
                post_bench = post.get("bench_ids", [])
                spidops_bench_idx = next((i for i, bid in enumerate(post_bench) if bid == 401), None)
                if spidops_bench_idx is not None and spidops_bench_idx < len(post_bench_energies):
                    pre_sp_e = pre_bench_energies[spidops_bench_idx] if spidops_bench_idx < len(pre_bench_energies) else 0
                    if post_bench_energies[spidops_bench_idx] > pre_sp_e:
                        r_strategic += 0.25
                    elif total_energy_gained > 0:
                        r_strategic += 0.15
                    else:
                        r_strategic -= 0.10
                elif total_energy_gained > 0:
                    r_strategic += 0.15
                else:
                    r_strategic -= 0.10

            elif played_id == 1216:
                pre_bench_size  = len(pre.get("bench_ids", []))
                post_bench_size = len(post.get("bench_ids", []))
                tr_mons_added   = post_bench_size - pre_bench_size
                hand_diff       = post["hand_size"] - pre["hand_size"]
                if tr_mons_added >= 2:
                    r_strategic += 0.30
                elif tr_mons_added == 1:
                    r_strategic += 0.20
                elif hand_diff >= 2:
                    r_strategic += 0.10
                else:
                    r_strategic -= 0.05

            elif played_id == 1220:
                hand_before = pre["hand_size"]
                hand_after  = post["hand_size"]
                hand_diff   = hand_after - hand_before
                if hand_before <= 3 and hand_diff >= 2:
                    r_strategic += 0.20
                elif hand_diff >= 2:
                    r_strategic += 0.10
                elif hand_before >= 6:
                    r_strategic -= 0.10
                else:
                    r_strategic += 0.05

            elif played_id == 1227:
                hand_before = pre["hand_size"]
                hand_after  = post["hand_size"]
                hand_diff   = hand_after - hand_before
                if hand_before <= 3 and hand_diff >= 2:
                    r_strategic += 0.20
                elif hand_diff >= 2:
                    r_strategic += 0.10
                elif hand_before >= 6:
                    r_strategic -= 0.08
                else:
                    r_strategic += 0.05

            elif played_id == 1219:
                pre_discard = pre.get("discard_ids", [])
                post_hand   = post.get("hand_ids", [])
                high_value_ids = [1216, 1218, 1220, 1217, 1227, 1129, 1116, 1097, 1159, 1175]
                recovered_high_value = any(cid in post_hand and cid in pre_discard for cid in high_value_ids)
                if recovered_high_value:
                    r_strategic += 0.25
                elif post["hand_size"] > pre["hand_size"]:
                    r_strategic += 0.10
                else:
                    r_strategic -= 0.05

            elif played_id == 1152:
                pre_discard = pre.get("discard_ids", [])
                supporter_ids_all = [1216, 1217, 1218, 1219, 1220, 1227]
                num_supporters_in_discard = sum(1 for cid in pre_discard if cid in supporter_ids_all)
                if num_supporters_in_discard >= 3:
                    r_strategic += 0.30
                elif num_supporters_in_discard >= 1:
                    r_strategic += 0.15
                else:
                    r_strategic -= 0.10

            elif played_id == 1129:
                pre_discard = pre.get("discard_ids", [])
                pokemon_ids_deck = [400, 401, 414, 431, 432, 434]
                num_pokemon_discard = sum(1 for cid in pre_discard if cid in pokemon_ids_deck)
                if num_pokemon_discard >= 4:
                    r_strategic += 0.40
                elif num_pokemon_discard >= 2:
                    r_strategic += 0.20
                else:
                    r_strategic -= 0.15

            elif played_id == 1218:
                opp_bench_count = len(pre.get("opp_bench_ids", []))
                if opp_bench_count > 0:
                    r_strategic += 0.20
                else:
                    r_strategic += 0.05

            elif played_id in [1216, 1227]:
                r_strategic += 0.15
            elif played_id in [1219, 1220]:
                r_strategic += 0.15
            elif played_id == 1134:
                r_strategic += 0.12
            elif played_id == 1094:
                r_strategic += 0.12

            elif played_card.cardType == CardType.SUPPORTER and played_id not in [1216, 1217, 1218, 1219, 1220, 1227]:
                hand_diff = post["hand_size"] - pre["hand_size"]
                if hand_diff < 0:
                    r_strategic -= 0.05
            else:
                r_strategic += 0.02
            
    elif action_type == 8:  # ATTACH
        attached_id = pre.get("attached_card_id", -1)
        target_id = pre.get("attached_target_id", -1)
        
        attached_card = get_card_data(attached_id)
        target_card = get_card_data(target_id)
        
        is_bench_attach = (pre.get("active_id") != target_id)
        if is_bench_attach and target_id != -1:
            target_bench_idx = pre.get("attached_target_bench_idx", -1)
            benched_energies = pre.get("bench_energies", [])
            if target_bench_idx >= 0 and target_bench_idx < len(benched_energies):
                current_bench_e = benched_energies[target_bench_idx]
            else:
                benched_ids = pre.get("bench_ids", [])
                try:
                    b_idx = benched_ids.index(target_id)
                    current_bench_e = benched_energies[b_idx] if b_idx < len(benched_energies) else 0
                except ValueError:
                    current_bench_e = 0
            
            if current_bench_e >= 2:
                r_strategic -= 0.30
        
        if attached_card is not None and target_card is not None:
            opp_active_card = get_card_data(pre.get("opp_active_id"))
            opp_is_ex = opp_active_card is not None and getattr(opp_active_card, "ex", False)
            opp_is_tera = opp_active_card is not None and (any(kw in opp_active_card.name.lower() for kw in ["dragapult", "tera"]) or getattr(opp_active_card, "terastal", False))
            charging_mimikyu = (target_id == 434 and opp_is_tera)
            if target_id == 434 and target_id in pre.get("bench_ids", []):
                if opp_is_tera:
                    r_strategic += 0.30
                else:
                    r_strategic -= 0.10

            target_is_blocker = is_defensive_blocker(target_card) and not charging_mimikyu
            if target_is_blocker:
                if attached_card.cardType == CardType.SPECIAL_ENERGY:
                    r_strategic -= 0.15
                else:
                    r_strategic -= 0.10
                    
            elif attached_card.cardType == CardType.SPECIAL_ENERGY:
                current_energy = 0
                if pre.get("active_id") == target_card.cardId:
                    current_energy = pre.get("active_energy", 0)
                else:
                    benched_ids = pre.get("bench_ids", [])
                    benched_energies = pre.get("bench_energies", [])
                    target_bench_idx = pre.get("attached_target_bench_idx", -1)
                    if target_bench_idx >= 0 and target_bench_idx < len(benched_energies):
                        current_energy = benched_energies[target_bench_idx]
                    else:
                        try:
                            b_idx = benched_ids.index(target_card.cardId)
                            if b_idx < len(benched_energies):
                                current_energy = benched_energies[b_idx]
                        except ValueError:
                            pass

                max_attack_cost = 0
                for aid in target_card.attacks:
                    att = attack_table.get(aid)
                    if att is not None:
                        max_attack_cost = max(max_attack_cost, len(att.energies))

                is_fully_charged = current_energy >= max_attack_cost

                if target_card.cardId in [431, 401, 434, 414]:
                    if is_fully_charged:
                        r_strategic -= 0.03
                    else:
                        r_strategic += 0.25
                        if pre.get("active_id") != target_card.cardId:
                            r_strategic += 0.08
                else:
                    r_strategic -= 0.05
                    
            elif attached_card.cardType == CardType.BASIC_ENERGY:
                current_energy = 0
                if pre.get("active_id") == target_card.cardId:
                    current_energy = pre.get("active_energy", 0)
                else:
                    benched_ids = pre.get("bench_ids", [])
                    benched_energies = pre.get("bench_energies", [])
                    target_bench_idx = pre.get("attached_target_bench_idx", -1)
                    if target_bench_idx >= 0 and target_bench_idx < len(benched_energies):
                        current_energy = benched_energies[target_bench_idx]
                    else:
                        try:
                            b_idx = benched_ids.index(target_card.cardId)
                            if b_idx < len(benched_energies):
                                current_energy = benched_energies[b_idx]
                        except ValueError:
                            pass

                max_attack_cost = 0
                for aid in target_card.attacks:
                    att = attack_table.get(aid)
                    if att is not None:
                        max_attack_cost = max(max_attack_cost, len(att.energies))

                is_fully_charged = current_energy >= max_attack_cost

                if target_card.cardId == 431:
                    if attached_card.cardId == 5:
                        if is_fully_charged:
                            r_strategic -= 0.03
                        else:
                            r_strategic += 0.25
                            if pre.get("active_id") != target_card.cardId:
                                r_strategic += 0.08
                    elif attached_card.cardId == 1:
                        if current_energy >= 2:
                            r_strategic -= 0.35
                        else:
                            r_strategic += 0.15
                elif target_card.cardId in [401, 400]:
                    if attached_card.cardId == 1:
                        if is_fully_charged:
                            r_strategic -= 0.03
                        else:
                            r_strategic += 0.25
                            if pre.get("active_id") != target_card.cardId:
                                r_strategic += 0.08
                    elif attached_card.cardId == 5:
                        r_strategic -= 0.03
                elif target_card.cardId == 434:
                    if attached_card.cardId == 5:
                        if is_fully_charged:
                            r_strategic -= 0.03
                        else:
                            r_strategic += 0.25
                            if pre.get("active_id") != target_card.cardId:
                                r_strategic += 0.08
                    elif attached_card.cardId == 1:
                        r_strategic -= 0.03
                elif target_card.cardId == 414:
                    if current_energy >= 1:
                        r_strategic -= 0.30
                    else:
                        r_strategic += 0.15
                else:
                    if is_fully_charged:
                        r_strategic -= 0.02
                    else:
                        target_is_attacker = target_card.ex or target_card.stage1 or target_card.stage2
                        if target_is_attacker:
                            r_strategic += 0.15
                            if pre.get("active_id") != target_card.cardId:
                                r_strategic += 0.05
                        else:
                            r_strategic += 0.05
                    
            elif attached_card.cardType == CardType.TOOL:
                opp_active_card = get_card_data(pre.get("opp_active_id"))
                is_opp_ex = opp_active_card is not None and (getattr(opp_active_card, "ex", False) or getattr(opp_active_card, "megaEx", False))
                
                if "Brave Bangle" in attached_card.name or "Maximum Belt" in attached_card.name:
                    target_is_heavy_attacker = target_card.cardId in [431, 401]
                    target_is_excluded = target_card.cardId in [434, 414]
                    
                    if target_is_heavy_attacker and is_opp_ex:
                        r_strategic += 0.25
                    elif target_is_excluded:
                        r_strategic -= 0.15
                    elif not target_is_heavy_attacker:
                        r_strategic -= 0.10
                    else:
                        r_strategic -= 0.05
                        
                elif "Hero" in attached_card.name or attached_card.cardId == 1159:
                    target_is_mewtwo = (target_card.cardId == 431)
                    is_active_target = (pre.get("active_id") == target_card.cardId)
                    target_is_defense = (target_card.cardId in [434, 401] and is_active_target)
                    
                    if target_is_mewtwo:
                        r_strategic += 0.30
                    elif target_is_defense:
                        r_strategic += 0.25
                    elif not is_active_target and target_card.cardId in [400, 414]:
                        r_strategic -= 0.40
                    elif target_card.cardId in [434]:
                        r_strategic -= 0.40
                    else:
                        r_strategic += 0.05
                         
    elif action_type == 9:  # EVOLVE
        evolved_id = pre.get("evolved_card_id", -1)
        evolved_card = get_card_data(evolved_id)
        if evolved_card is not None:
            if evolved_id == 401 or getattr(evolved_card, "cardId", -1) == 401:
                r_strategic += 0.35
                if pre.get("active_id") == 400:
                    r_strategic += 0.15
                if pre.get("turn", 1) <= 5:
                    r_strategic += 0.15
            elif evolved_card.stage1 or evolved_card.stage2:
                r_strategic += 0.20
            else:
                r_strategic += 0.10
                
    elif action_type == 10:  # ABILITY
        r_strategic += 0.20
        
    elif action_type == 12:  # RETREAT
        r_strategic -= 0.40
        opp_active_card = get_card_data(pre.get("opp_active_id"))
        active_card = get_card_data(pre.get("active_id"))
        post_active_card = get_card_data(post.get("active_id"))
        active_id_pre = pre.get("active_id", -1)
        active_energies = pre.get("active_energies", 0)
        has_attack_option = pre.get("has_attack_option", False)
        
        RETREAT_COSTS = {
            434: 0,
            400: 1,
            414: 1,
            401: 2,
            431: 3,
        }
        retreat_cost = RETREAT_COSTS.get(active_id_pre, 1)
        energy_discarded = min(active_energies, retreat_cost)
        
        if retreat_cost > 0 and energy_discarded > 0 and has_attack_option:
            if active_id_pre == 401:
                r_strategic -= 0.40 * energy_discarded
            elif active_id_pre == 431:
                pass
            else:
                r_strategic -= 0.15 * energy_discarded
        
        opp_has_immunity = opp_active_card is not None and (
            opp_active_card.cardId == 345 or 
            any(getattr(s, "name", "") == "Safeguard" or "ex" in getattr(s, "text", "").lower() for s in getattr(opp_active_card, "skills", []))
        )
        
        if opp_has_immunity:
            if active_card is not None and active_card.ex and post_active_card is not None and not post_active_card.ex:
                r_strategic += 0.35
            elif post_active_card is not None and is_defensive_blocker(post_active_card):
                r_strategic += 0.20
        elif active_card is not None and active_card.ex and pre.get("active_energies", 0) >= 3:
            r_strategic -= 0.05
            if active_card.cardId == 431:
                opp_hp = pre.get("opp_active_hp", 999)
                if opp_hp <= 280:
                    r_strategic -= 0.80
                    
        active_hp_pre = pre.get("active_hp", 999)
        opp_has_bench_snipe = (opp_active_card is not None and (
            opp_active_card.cardId in [648, 121] or 
            any(kw in opp_active_card.name.lower() for kw in ["grimmsnarl", "dragapult", "shrouded", "bench"])
        ))
        
        if active_hp_pre <= 30 and opp_has_bench_snipe:
            r_strategic -= 0.50
            
        if post_active_card is not None and post_active_card.cardId == 434:
            opp_is_darkness = opp_active_card is not None and (getattr(opp_active_card, "energyType", None) == EnergyType.DARKNESS or "grimmsnarl" in opp_active_card.name.lower())
            if opp_is_darkness:
                r_strategic -= 0.50
            elif opp_active_card is not None and any(kw in opp_active_card.name.lower() for kw in ["dragapult", "tera"]):
                if pre.get("active_energies", 0) >= 2:
                    r_strategic += 0.30
            
        if active_card is not None and active_card.cardId == 431:
            if pre.get("active_hp", 280) <= 140:
                if post_active_card is not None and is_defensive_blocker(post_active_card):
                    r_strategic += 0.50
            
    elif action_type == 14:  # END (Turn Stall Decisions)
        has_attack = pre.get("has_attack_option", False)
        has_attach = pre.get("has_attach_option", False)
        energy_already_attached = pre.get("energy_attached_flag", False)
        
        opp_active_card = get_card_data(pre.get("opp_active_id"))
        active_card = get_card_data(pre.get("active_id"))
        opp_immune = (opp_active_card is not None and active_card is not None and active_card.ex and (
            opp_active_card.cardId == 345 or 
            any(getattr(s, "name", "") == "Safeguard" or "ex" in getattr(s, "text", "").lower() for s in getattr(opp_active_card, "skills", []))
        ))
        
        if has_attack and not opp_immune:
            r_strategic -= 0.40
        elif has_attach and not energy_already_attached:
            r_strategic -= 0.40
        else:
            r_strategic += 0.02
            
        if post.get("bench_size", 0) == 0:
            r_strategic -= 0.60
            
        has_basic_in_hand = any(cid in [400, 414, 431, 434, 432] for cid in post.get("hand_ids", []))
        bench_size_post = post.get("bench_size", 0)
        if has_basic_in_hand and bench_size_post < 5:
            r_strategic -= 0.40
            
    elif action_type == 13:  # ATTACK
        opp_active_card = get_card_data(pre.get("opp_active_id"))
        active_card = get_card_data(pre.get("active_id"))
        opp_immune = (opp_active_card is not None and active_card is not None and active_card.ex and (
            opp_active_card.cardId == 345 or 
            any(getattr(s, "name", "") == "Safeguard" or "ex" in getattr(s, "text", "").lower() for s in getattr(opp_active_card, "skills", []))
        ))
        
        if opp_immune:
            r_strategic -= 0.60
        else:
            if opp_active_card is not None and opp_active_card.cardId == 345 and active_card is not None and active_card.cardId == 401:
                r_strategic += 0.40
            
            att_id = pre.get("attack_id", -1)
            if att_id == 560:
                tr_count_pre = 0
                active_id_pre = pre.get("active_id", -1)
                bench_ids_pre = pre.get("bench_ids", [])
                if is_tr_pokemon(active_id_pre):
                    tr_count_pre += 1
                for bid in bench_ids_pre:
                    if is_tr_pokemon(bid):
                        tr_count_pre += 1
                
                r_strategic += 0.05 + 0.08 * (tr_count_pre - 2) if tr_count_pre >= 2 else 0.02
            elif att_id == 609:
                pre_dmg = sum(pre.get("bench_damage", []))
                post_dmg = sum(post.get("bench_damage", []))
                healed = pre_dmg - post_dmg
                if healed > 0:
                    r_strategic += 0.02 * healed
            elif active_card is not None and active_card.cardId == 431:
                tr_count_mewtwo = 0
                if is_tr_pokemon(pre.get("active_id", -1)):
                    tr_count_mewtwo += 1
                for bid in pre.get("bench_ids", []):
                    if is_tr_pokemon(bid):
                        tr_count_mewtwo += 1
                if tr_count_mewtwo >= 4:
                    r_strategic += 0.35
                else:
                    r_strategic -= 0.20
            elif active_card is not None and active_card.cardId == 414:
                r_strategic += 0.30
                if pre.get("turn", 0) <= 3:
                    r_strategic += 0.15
            else:
                r_strategic += 0.20
            if not pre.get("has_attacked_flag", False):
                r_strategic += 0.10

            prizes_taken = pre["prizes"] - post["prizes"]
            if pre.get("turn", 0) <= 3 and prizes_taken > 0:
                r_strategic += 0.35
            
            opp_active_hp_pre = pre.get("opp_active_hp", 999)
            opp_active_max_hp = pre.get("opp_active_max_hp", 999)
            opp_prizes_pre = pre.get("prizes_remaining_opp", 0)
            turn_pre = pre.get("turn", 99)
            if opp_active_max_hp <= 100 and turn_pre <= 10 and opp_prizes_pre >= 4:
                r_strategic += 0.20
            
            opp_active_id_pre = pre.get("opp_active_id", -1)
            opp_bench_ids_pre = pre.get("opp_bench_ids", [])
            GRIMMSNARL_EX_ID_TRAIN = 648
            MUNKIDORI_ID_TRAIN = 112
            is_munkidori_active = (opp_active_id_pre == MUNKIDORI_ID_TRAIN)
            is_grimmsnarl_on_bench = (GRIMMSNARL_EX_ID_TRAIN in opp_bench_ids_pre)
            if is_munkidori_active and is_grimmsnarl_on_bench:
                r_strategic += 0.25

    if action_type == 8:
        attached_cid = pre.get("attached_card_id", -1)
        target_cid = pre.get("attached_target_id", -1)
        if (attached_cid == 15 or pre.get("card_id") == 15) and (target_cid == 414 or pre.get("target_id") == 414):
            r_strategic += 0.20

    if action_type == 12:
        active_id_pre = pre.get("active_id", -1)
        if active_id_pre == 434:
            r_strategic += 0.10

    if action_type == 11:
        active_id_pre = pre.get("active_id", -1)
        if active_id_pre == 401:
            r_strategic += 0.10

    if action_type in (6, 7):
        played_cid = pre.get("played_card_id", -1)
        if pre.get("bench_size", 0) == 0 and (played_cid in (1134, 1086, 1220) or pre.get("card_id") in (1134, 1086, 1220)):
            r_strategic += 0.20
        elif (played_cid == 1218 or pre.get("card_id") == 1218):
            opp_bench_count = len(pre.get("opp_bench_ids", []))
            if opp_bench_count > 0:
                r_strategic += 0.20
        elif played_cid == 1159 or pre.get("card_id") == 1159:
            r_strategic += 0.20
        elif played_cid == 1220 or pre.get("card_id") == 1220:
            if pre.get("turn", 1) <= 1:
                r_strategic += 0.30
            elif pre.get("bench_size", 5) < 3:
                r_strategic += 0.15
            else:
                r_strategic += 0.10
        elif played_cid == 1227 or pre.get("card_id") == 1227:
            my_deck_size = pre.get("deck_size", 40)
            my_hand_size = pre.get("hand_size", 5)
            hand_diff = post.get("hand_size", 5) - pre.get("hand_size", 5)
            if my_deck_size <= 5:
                r_strategic -= 0.15
            elif my_hand_size <= 3:
                r_strategic += 0.15
            elif hand_diff >= 2:
                r_strategic += 0.10
            elif my_hand_size >= 6 and hand_diff <= 0:
                r_strategic -= 0.05

        TR_SUPPORTER_CDS = {1216, 1218, 1219, 1220}
        played_cid_now = played_cid if played_cid != -1 else pre.get("card_id", -1)
        if played_cid_now in TR_SUPPORTER_CDS:
            factory_in_play = (pre.get("stadium_id", -1) == 1257)
            if factory_in_play:
                r_strategic += 0.15

        if played_cid_now == 414:
            bench_size_pre = pre.get("bench_size", 5)
            if bench_size_pre < 5:
                r_strategic += 0.10

    if player_idx == 0:
        opp_active_now = pre.get("opp_active_id", -1)
        my_active_now = pre.get("active_id", -1)
        prizes_taken = pre["prizes"] - post["prizes"]
        opp_deck_size = pre.get("opp_deck_size", 40)
        
        if opponent_name == "Rulebasedmodel_Abomasnow":
            if prizes_taken > 0 and opp_active_now == 722:
                r_strategic += 0.35
            if action_type == 13 and opp_active_now in [721, 722]:
                r_strategic += 0.15
            if opp_active_now == 723:
                if my_active_now == 434:
                    r_strategic -= 0.10
                elif my_active_now == 401:
                    if pre.get("active_energies", 0) >= 2:
                        r_strategic += 0.15
                        if action_type == 13:
                            r_strategic += 0.20
                elif my_active_now == 431:
                    r_strategic -= 0.10
                elif my_active_now == 400:
                    r_strategic -= 0.10
                
                attached_t_id = pre.get("attached_target_id", -1)
                if action_type == 8 and (attached_t_id in [400, 401] or pre.get("target_id") in [400, 401]):
                    r_strategic += 0.10
                if action_type == 8 and (attached_t_id == 434 or pre.get("target_id") == 434):
                    r_strategic += 0.10
            if opp_deck_size <= 10 and action_type == 14:
                r_strategic += 0.15
                    
        elif opponent_name == "Rulebasedmodel_Dragapult":
            if prizes_taken > 0 and opp_active_now in [119, 120]:
                r_strategic += 0.35
            if opp_active_now == 121:
                if my_active_now == 434:
                    if pre.get("active_energies", 0) >= 2:
                        r_strategic += 0.15
                        if action_type == 13:
                            r_strategic += 0.20
                    else:
                        r_strategic += 0.10
                elif my_active_now == 431:
                    r_strategic -= 0.10
                attached_t_id = pre.get("attached_target_id", -1)
                if action_type == 8 and (attached_t_id == 434 or pre.get("target_id") == 434):
                    r_strategic += 0.15
                attached_c_id = pre.get("attached_card_id", -1)
                if action_type == 8 and (attached_c_id == 1159 or pre.get("card_id") == 1159):
                    r_strategic += 0.15
                    
                b_size = post.get("bench_size", 0)
                if b_size > 2:
                    r_strategic -= 0.20 * (b_size - 2)

        elif opponent_name == "Rulebasedmodel_Iono":
            played_c_id = pre.get("played_card_id", -1)
            if action_type == 6 and (played_c_id in [1134, 1216, 1257] or pre.get("card_id") in [1134, 1216, 1257]):
                if pre.get("hand_size", 0) <= 3:
                    r_strategic += 0.25
                else:
                    r_strategic += 0.15
            if prizes_taken > 0 and opp_active_now in [265, 268, 269, 270, 271]:
                r_strategic += 0.20

    return r_strategic
