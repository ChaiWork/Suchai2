import sys
import random
import torch
from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_observation_class, OptionType, SelectContext, CardType, EnergyType

from model import (
    MyModel,
    MODEL_D_MODEL,
    MODEL_NUM_HEADS,
    MODEL_D_FEEDFORWARD,
    MODEL_NUM_LAYERS_ENCODER,
    MODEL_NUM_LAYERS_DECODER,
)
from agent import mcts_agent, random_agent, GPUInferenceClient
try:
    from expert_knowledge import get_expert_bonus
except ImportError:
    from src.expert_knowledge import get_expert_bonus

from training.card_database import (
    get_card_data, is_defensive_blocker, can_attack,
    count_effective_energy_cards, count_attached_energy,
    count_active_energy, count_pokemon, attack_table
)
from training.evaluator import rule_based_opponent_agent
from training.gae import compute_gae_advantages
from training.rewards import calculate_strategic_reward


def worker_loop(worker_id, command_queue, result_queue, inference_conn, device_str):
    """The game worker execution loop."""
    random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)
    device = torch.device(device_str)
    
    client = GPUInferenceClient(inference_conn)
    
    while True:
        try:
            cmd, args = command_queue.get()
        except KeyboardInterrupt:
            break
            
        if cmd == "STOP":
            break
            
        elif cmd == "PLAY_SELF":
            sample_deck, opponent_deck, opponent_type, opponent_path = args[:4]
            opponent_name = args[4] if len(args) > 4 else "Current (Self)"
            current_epoch = args[5] if len(args) > 5 else 0  # epoch counter for warmup schedule
            
            opp_model = None
            if opponent_path is not None:
                try:
                    _opp_model_module = MyModel(
                        MODEL_D_MODEL,
                        MODEL_NUM_HEADS,
                        MODEL_D_FEEDFORWARD,
                        MODEL_NUM_LAYERS_ENCODER,
                        MODEL_NUM_LAYERS_DECODER
                    ).to(device)
                    checkpoint = torch.load(opponent_path, map_location=device, weights_only=True)
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        _opp_model_module.load_state_dict(checkpoint["state_dict"], strict=False)
                    else:
                        _opp_model_module.load_state_dict(checkpoint, strict=False)
                    _opp_model_module.eval()
                    opp_model = GPUInferenceClient(_opp_model_module, device)
                except Exception as e:
                    opp_model = None
            
            try:
                obs, start_data = battle_start(sample_deck, opponent_deck)
                if start_data.errorPlayer >= 0:
                    result_queue.put(("PLAY_SELF_COMPLETE", (worker_id, [], -1, 0, {}, 0.0, 0, {}, {})))
                    continue
            except Exception as e:
                result_queue.put(("PLAY_SELF_COMPLETE", (worker_id, [], -1, 0, {}, 0.0, 0, {}, {})))
                continue

            try:
                action_counts = {"attack": 0, "play": 0, "attach": 0, "evolve": 0, "ability": 0, "retreat": 0, "end": 0, "other": 0}
                played_cards = {}
                samples = [[], []]
                expert_log = []  # Per-step expert guidance log
                
                # Episode-state active spot lockout trackers
                episode_lockouts = [0, 0]
                episode_active_serials = [None, None]
                episode_attacked = [False, False]
                episode_last_turns = [None, None]
                
                step_count = 0
                while True:
                    step_count += 1
                    if obs["current"]["result"] >= 0 or step_count > 400:
                        if step_count > 400 and obs["current"]["result"] < 0:
                            obs["current"]["result"] = 2  # Force game draw on timeout to prevent infinite worker hangs
                        break
    
                    curr_player = obs["current"]["yourIndex"]
                    curr_deck = sample_deck if curr_player == 0 else opponent_deck
                    
                    obs_class = to_observation_class(obs)
                    state_ps = obs_class.current.players[curr_player]
                    opp_ps = obs_class.current.players[1 - curr_player]
                    
                    active_pk = state_ps.active[0] if (len(state_ps.active) > 0 and state_ps.active[0] is not None) else None
                    active_id = active_pk.id if active_pk else -1
                    active_energies = count_effective_energy_cards(active_pk.energyCards) if active_pk else 0
                    
                    # Track active spot lockout at the episode level
                    active_serial = active_pk.serial if active_pk else None
                    turn_num = obs_class.current.turn
                    
                    if episode_active_serials[curr_player] != active_serial:
                        episode_active_serials[curr_player] = active_serial
                        episode_lockouts[curr_player] = 0
                        episode_attacked[curr_player] = False
                        
                    if episode_last_turns[curr_player] is not None and turn_num != episode_last_turns[curr_player]:
                        if not episode_attacked[curr_player]:
                            episode_lockouts[curr_player] += 1
                        episode_attacked[curr_player] = False
                    episode_last_turns[curr_player] = turn_num
                    
                    opp_active_pk = opp_ps.active[0] if (len(opp_ps.active) > 0 and opp_ps.active[0] is not None) else None
                    opp_active_id = opp_active_pk.id if opp_active_pk else -1
                    
                    bench_list = [p for p in state_ps.bench if p is not None]
                    bench_ids = [p.id for p in bench_list]
                    
                    has_attack_option = False
                    has_attach_option = False
                    if obs_class.select is not None and obs_class.select.option is not None:
                        for opt in obs_class.select.option:
                            if opt.type == OptionType.ATTACK:
                                has_attack_option = True
                            elif opt.type == OptionType.ATTACH:
                                has_attach_option = True

                    stadium_id = obs_class.current.stadium[0].id if (len(obs_class.current.stadium) > 0 and obs_class.current.stadium[0] is not None) else -1
                    
                    pre_metrics = {
                        "prizes": len(state_ps.prize),
                        "opp_prizes": len(opp_ps.prize),
                        "energy": count_attached_energy(state_ps),
                        "active_energy": count_active_energy(state_ps),
                        "pokemon": count_pokemon(state_ps),
                        "opp_pokemon": count_pokemon(opp_ps),
                        "opp_bench_ids": [p.id for p in opp_ps.bench if p is not None],
                        "bench_size": len(bench_list),
                        "deck_size": state_ps.deckCount,
                        "opp_deck_size": opp_ps.deckCount,
                        "energy_attached_flag": obs_class.current.energyAttached,
                        "active_id": active_id,
                        "active_energies": active_energies,
                        "opp_active_id": opp_active_id,
                        "active_hp": active_pk.hp if active_pk else 0,
                        "opp_active_hp": opp_active_pk.hp if opp_active_pk else 0,
                        "stadium_id": stadium_id,
                        "bench_ids": bench_ids,
                        "bench_energies": [count_effective_energy_cards(p.energyCards) for p in state_ps.bench if p is not None],
                        "bench_damage": [p.maxHp - p.hp for p in state_ps.bench if p is not None],
                        "hand_size": len(state_ps.hand) if state_ps.hand is not None else 0,
                        "hand_ids": [c.id for c in state_ps.hand if c is not None] if state_ps.hand is not None else [],
                        "discard_size": len(state_ps.discard) if state_ps.discard is not None else 0,
                        "discard_energy": sum(1 for c in state_ps.discard if c is not None and c.id in [1, 5, 15]),
                        "turn": obs_class.current.turn,
                        "has_attack_option": has_attack_option,
                        "has_attach_option": has_attach_option,
                        "context": obs_class.select.context if (obs_class.select is not None) else None,
                        "lockout_turns": episode_lockouts[curr_player]
                    }
    
                    if curr_player == 0:
                        turn = obs_class.current.turn if (obs_class.current is not None) else 1
                        warmup_threshold = max(0, 5 - (current_epoch // 5))
                        
                        use_warmup = (turn <= warmup_threshold)
                        if use_warmup:
                            try:
                                rb_selected = rule_based_opponent_agent("Rulebasedmodel_Mewtwo", obs)
                                temperature = 1.0 if turn <= 15 else 0.1
                                selected, sample = mcts_agent(obs, curr_deck, client, search_count=50, temperature=temperature, force_action=rb_selected, opponent_name=opponent_name, epoch=current_epoch)
                            except Exception as e:
                                use_warmup = False
                                
                        if not use_warmup:
                            temperature = 1.0 if turn <= 15 else 0.1
                            try:
                                selected, sample = mcts_agent(obs, curr_deck, client, search_count=50, temperature=temperature, opponent_name=opponent_name, epoch=current_epoch)
                            except Exception as mcts_err:
                                selected = [0]
                                sample = None
                            if sample is not None:
                                sample.pred_val = sample.value
                            
                            if selected and len(selected) > 0:
                                try:
                                    chosen_opt = obs_class.select.option[selected[0]]
                                    exp_bonus, exp_trigger = get_expert_bonus(obs_class, chosen_opt, opponent_name=opponent_name)
                                    if exp_trigger and str(exp_trigger).lower() not in ("none", "disabled") and abs(exp_bonus) > 1e-6:
                                        expert_log.append({
                                            "trigger":     exp_trigger,
                                            "action_type": getattr(chosen_opt, "type", getattr(chosen_opt, "optionType", -1)),
                                            "bonus":       exp_bonus,
                                            "turn":        obs_class.current.turn,
                                            "my_prizes":   len(obs_class.current.players[curr_player].prize),
                                            "opp_prizes":  len(obs_class.current.players[1 - curr_player].prize),
                                        })
                                except Exception:
                                    pass
                            
                            opt_type_val = -1
                            played_card_id = -1
                            attached_card_id = -1
                            attached_target_id = -1
                            evolved_card_id = -1
                            attack_id = -1
                            if selected and len(selected) > 0:
                                sel_idx = selected[0]
                                options = obs.get("select", {}).get("option", [])
                                if sel_idx < len(options):
                                    opt = options[sel_idx]
                                    opt_type_val = opt.get("type", -1)
                                    if opt_type_val == 7: # PLAY
                                        hand = state_ps.hand
                                        card_idx = opt.get("index", -1)
                                        if 0 <= card_idx < len(hand) and hand[card_idx] is not None:
                                            played_card_id = hand[card_idx].id
                                            played_cards[played_card_id] = played_cards.get(played_card_id, 0) + 1
                                    elif opt_type_val == 8: # ATTACH
                                        hand = state_ps.hand
                                        card_idx = opt.get("index", -1)
                                        if 0 <= card_idx < len(hand) and hand[card_idx] is not None:
                                            attached_card_id = hand[card_idx].id
                                        target_area = opt.get("inPlayArea", -1)
                                        target_idx = opt.get("inPlayIndex", -1)
                                        if target_area == 4: # ACTIVE
                                            if len(state_ps.active) > 0 and state_ps.active[0] is not None:
                                                attached_target_id = state_ps.active[0].id
                                        elif target_area == 5: # BENCH
                                            if 0 <= target_idx < len(state_ps.bench) and state_ps.bench[target_idx] is not None:
                                                attached_target_id = state_ps.bench[target_idx].id
                                    elif opt_type_val == 9: # EVOLVE
                                        hand = state_ps.hand
                                        card_idx = opt.get("index", -1)
                                        if 0 <= card_idx < len(hand) and hand[card_idx] is not None:
                                            evolved_card_id = hand[card_idx].id
                                    elif opt_type_val == 13: # ATTACK
                                        attack_id = opt.get("attackId", -1)
                                        
                            pre_metrics["action_type"] = opt_type_val
                            pre_metrics["played_card_id"] = played_card_id
                            pre_metrics["attached_card_id"] = attached_card_id
                            pre_metrics["attached_target_id"] = attached_target_id
                            pre_metrics["evolved_card_id"] = evolved_card_id
                            pre_metrics["attack_id"] = attack_id
                            samples[0].append((sample, pre_metrics))

                        if selected and len(selected) > 0:
                            sel_idx = selected[0]
                            options = obs.get("select", {}).get("option", [])
                            if sel_idx < len(options):
                                opt_type = options[sel_idx].get("type")
                                if opt_type == 13:
                                    action_counts["attack"] += 1
                                elif opt_type == 7:
                                    action_counts["play"] += 1
                                elif opt_type == 8:
                                    action_counts["attach"] += 1
                                elif opt_type == 9:
                                    action_counts["evolve"] += 1
                                elif opt_type == 10:
                                    action_counts["ability"] += 1
                                elif opt_type == 12:
                                    action_counts["retreat"] += 1
                                elif opt_type == 14:
                                    action_counts["end"] += 1
                                else:
                                    action_counts["other"] += 1
                    else:
                        if opponent_name.startswith("Rulebasedmodel"):
                            try:
                                selected = rule_based_opponent_agent(opponent_name, obs)
                            except Exception as e:
                                selected = random_agent(obs)
                        elif opponent_type == "Current":
                            opp_temperature = 1.0 if turn_num <= 15 else 0.1
                            selected, sample = mcts_agent(obs, curr_deck, client, search_count=50, temperature=opp_temperature, epoch=current_epoch)
                            sample.pred_val = sample.value
                            
                            opt_type_val = -1
                            played_card_id = -1
                            attached_card_id = -1
                            attached_target_id = -1
                            evolved_card_id = -1
                            attack_id = -1
                            if selected and len(selected) > 0:
                                sel_idx = selected[0]
                                options = obs.get("select", {}).get("option", [])
                                if sel_idx < len(options):
                                    opt = options[sel_idx]
                                    opt_type_val = opt.get("type", -1)
                                    if opt_type_val == 7: # PLAY
                                        hand = state_ps.hand
                                        card_idx = opt.get("index", -1)
                                        if 0 <= card_idx < len(hand) and hand[card_idx] is not None:
                                            played_card_id = hand[card_idx].id
                                    elif opt_type_val == 8: # ATTACH
                                        hand = state_ps.hand
                                        card_idx = opt.get("index", -1)
                                        if 0 <= card_idx < len(hand) and hand[card_idx] is not None:
                                            attached_card_id = hand[card_idx].id
                                        target_area = opt.get("inPlayArea", -1)
                                        target_idx = opt.get("inPlayIndex", -1)
                                        if target_area == 4: # ACTIVE
                                            if len(state_ps.active) > 0 and state_ps.active[0] is not None:
                                                attached_target_id = state_ps.active[0].id
                                        elif target_area == 5: # BENCH
                                            if 0 <= target_idx < len(state_ps.bench) and state_ps.bench[target_idx] is not None:
                                                attached_target_id = state_ps.bench[target_idx].id
                                    elif opt_type_val == 9: # EVOLVE
                                        hand = state_ps.hand
                                        card_idx = opt.get("index", -1)
                                        if 0 <= card_idx < len(hand) and hand[card_idx] is not None:
                                            evolved_card_id = hand[card_idx].id
                                    elif opt_type_val == 13: # ATTACK
                                        attack_id = opt.get("attackId", -1)
                                        
                            pre_metrics["action_type"] = opt_type_val
                            pre_metrics["played_card_id"] = played_card_id
                            pre_metrics["attached_card_id"] = attached_card_id
                            pre_metrics["attached_target_id"] = attached_target_id
                            pre_metrics["evolved_card_id"] = evolved_card_id
                            pre_metrics["attack_id"] = attack_id
                            samples[1].append((sample, pre_metrics))
                        elif opp_model is not None:
                            selected, sample = mcts_agent(obs, curr_deck, opp_model, search_count=50)
                        else:
                            selected = random_agent(obs)
                    
                    if selected and len(selected) > 0:
                        sel_idx = selected[0]
                        options = obs.get("select", {}).get("option", [])
                        if sel_idx < len(options):
                            opt = options[sel_idx]
                            opt_type = opt.get("type", -1)
                            if opt_type == 13 or opt_type == OptionType.ATTACK:
                                episode_attacked[curr_player] = True
                                
                    try:
                        obs = battle_select(selected)
                    except IndexError:
                        selected = random_agent(obs)
                        obs = battle_select(selected)
                    
                battle_finish()
            except Exception as e:
                import traceback
                print(f"Error in worker {worker_id} simulation:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                result_queue.put(("PLAY_SELF_COMPLETE", (worker_id, [], -1, 0, {}, 0.0, 0, {}, {})))
                continue
            
            result = obs["current"]["result"]
            obs_class = to_observation_class(obs)
            final_turn = obs_class.current.turn if (obs_class is not None and obs_class.current is not None) else 0
            
            processed_samples = []
            rc_worker = {"prize_taken": 0.0, "prize_lost": 0.0, "kos": 0.0, "own_kos": 0.0,
                         "energy": 0.0, "bench": 0.0, "deckout": 0.0, "terminal": 0.0, "stall": 0.0,
                         "no_energy": 0.0, "strategic": 0.0}
            
            for i in range(2):
                player_samples = samples[i]
                n_steps = len(player_samples)
                if n_steps == 0:
                    continue
                    
                first_turn_of_player = player_samples[0][1].get("turn", 1)
                went_second = (first_turn_of_player == 2)
                
                num_attacks = sum(1 for _, pre in player_samples if pre.get("action_type") == 13)
                if i == result:
                    terminal_reward = 1.0
                elif result == 2:
                    terminal_reward = 0.0
                elif result == -1:
                    terminal_reward = -1.0
                else:
                    terminal_reward = -1.0

                rewards = []
                has_attacked_flag = False
                has_taken_prize_flag = False
                has_lost_prize_flag = False
                last_mimikyu_retreat_turn = -999
                for step_idx in range(n_steps):
                    sample_obj, pre = player_samples[step_idx]
                    
                    if step_idx < n_steps - 1:
                        _, post = player_samples[step_idx + 1]
                    else:
                        final_obs = to_observation_class(obs)
                        final_ps = final_obs.current.players[i]
                        final_opp_ps = final_obs.current.players[1 - i]
                        final_active_pk = final_ps.active[0] if (len(final_ps.active) > 0 and final_ps.active[0] is not None) else None
                        final_active_id = final_active_pk.id if final_active_pk else -1
                        final_active_energies = count_effective_energy_cards(final_active_pk.energyCards) if final_active_pk else 0
                        
                        final_opp_active_pk = final_opp_ps.active[0] if (len(final_opp_ps.active) > 0 and final_opp_ps.active[0] is not None) else None
                        final_opp_active_id = final_opp_active_pk.id if final_opp_active_pk else -1
                        
                        final_bench_list = [p for p in final_ps.bench if p is not None]
                        final_bench_ids = [p.id for p in final_bench_list]
                        final_stadium_id = final_obs.current.stadium[0].id if (len(final_obs.current.stadium) > 0 and final_obs.current.stadium[0] is not None) else -1
                        
                        post = {
                            "prizes": len(final_ps.prize),
                            "opp_prizes": len(final_opp_ps.prize),
                            "energy": count_attached_energy(final_ps),
                            "active_energy": count_active_energy(final_ps),
                            "pokemon": count_pokemon(final_ps),
                            "opp_pokemon": count_pokemon(final_opp_ps),
                            "opp_bench_ids": [p.id for p in final_opp_ps.bench if p is not None],
                            "bench_size": len(final_bench_list),
                            "deck_size": final_ps.deckCount,
                            "opp_deck_size": final_opp_ps.deckCount,
                            "energy_attached_flag": True,
                            "active_id": final_active_id,
                            "active_energies": final_active_energies,
                            "opp_active_id": final_opp_active_id,
                            "active_hp": final_active_pk.hp if final_active_pk else 0,
                            "opp_active_hp": final_opp_active_pk.hp if final_opp_active_pk else 0,
                            "stadium_id": final_stadium_id,
                            "bench_ids": final_bench_ids,
                            "bench_energies": [count_effective_energy_cards(p.energyCards) for p in final_ps.bench if p is not None],
                            "bench_damage": [p.maxHp - p.hp for p in final_ps.bench if p is not None],
                            "hand_size": len(final_ps.hand) if final_ps.hand is not None else 0,
                            "hand_ids": [c.id for c in final_ps.hand if c is not None] if final_ps.hand is not None else [],
                            "discard_size": len(final_ps.discard) if final_ps.discard is not None else 0,
                            "discard_energy": sum(1 for c in final_ps.discard if c is not None and c.id in [1, 5, 15]),
                            "turn": final_obs.current.turn,
                            "has_attack_option": False,
                            "has_attach_option": False,
                            "lockout_turns": episode_lockouts[i]
                        }
                    
                    prizes_taken = pre["prizes"] - post["prizes"]
                    prizes_lost = pre["opp_prizes"] - post["opp_prizes"]
                    opp_kos = pre["opp_pokemon"] - post["opp_pokemon"]
                    own_kos = pre["pokemon"] - post["pokemon"]
                    energy_attached = post["energy"] - pre["energy"]
                    active_energy_attached = post.get("active_energy", 0) - pre.get("active_energy", 0)
                    bench_energy_attached = energy_attached - active_energy_attached
                    
                    lockout_turns = pre.get("lockout_turns", 0)
                    opp_deck_size = pre.get("opp_deck_size", 40)
                    if opp_deck_size <= 5:
                        r_stall = 0.0
                    else:
                        r_stall = -0.02
                        if lockout_turns > 3:
                            r_stall -= min(0.20, 0.02 * (lockout_turns - 3))
                        
                    r_prize_t = prizes_taken * 0.5 if prizes_taken > 0 else 0.0
                    if prizes_taken > 0 and not has_taken_prize_flag:
                        r_prize_t += 0.50
                        has_taken_prize_flag = True
                    r_prize_l = - (prizes_lost * 0.35) if prizes_lost > 0 else 0.0
                    if prizes_lost > 0 and not has_lost_prize_flag:
                        r_prize_l -= 0.35
                        has_lost_prize_flag = True
                    r_ko = 0.0
                    r_own_ko = - (own_kos * 0.15) if own_kos > 0 else 0.0
                    
                    action_type = pre.get("action_type", -1)
                    r_en = 0.0
                    if energy_attached > 0:
                        attached_target = pre.get("attached_target_id", -1)
                        target_card = get_card_data(attached_target) if attached_target > 0 else None
                        max_req_en = 2
                        if target_card and hasattr(target_card, "attacks"):
                            costs = [len(attack_table[aid].energies) for aid in getattr(target_card, "attacks", []) if aid in attack_table]
                            if costs:
                                max_req_en = max(costs)
                        target_curr_en = pre.get("active_energies", 0) if attached_target == pre.get("active_id") else 0
                        if target_curr_en >= max_req_en:
                            r_en = -0.25
                        else:
                            r_en = 0.05
                    r_no_en = -0.05 if (action_type != 8 and pre.get("has_energy_in_hand", False)) else 0.0
                        
                    r_strategic = calculate_strategic_reward(pre, post, action_type, step_idx, went_second, i, opponent_name)

                    r_bench = 0.0
                    if post["bench_size"] == 0:
                        r_bench -= 0.05
                        if post.get("turn", 0) <= 2:
                            r_bench -= 0.05
                    elif post["bench_size"] >= 3 and any(cid in (65, 66, 67) for cid in post.get("opp_bench_ids", []) + [post.get("opp_active_id", -1)]):
                        r_bench -= 0.05
                    elif post["bench_size"] == 5:
                        r_bench -= 0.02

                    r_deck = 0.0
                    if post["deck_size"] == 0:
                        r_deck -= 0.50
                        
                    STRATEGIC_SCALE = 0.05
                    step_reward = r_stall + r_prize_t + r_prize_l + r_ko + r_own_ko + r_en + r_no_en + r_bench + r_deck + (r_strategic * STRATEGIC_SCALE)
                    step_reward = max(-0.15, min(0.15, step_reward))

                    if i == 0:
                        rc_worker["prize_taken"] += r_prize_t
                        rc_worker["prize_lost"] += r_prize_l
                        rc_worker["kos"] += prizes_taken
                        rc_worker["own_kos"] += r_own_ko
                        rc_worker["energy"] += r_en
                        rc_worker["no_energy"] += r_no_en
                        rc_worker["bench"] += r_bench
                        rc_worker["deckout"] += r_deck
                        rc_worker["stall"] += r_stall
                        rc_worker["strategic"] += r_strategic
                        
                    rewards.append(step_reward)
                    
                if i == 0:
                    rc_worker["terminal"] += terminal_reward
                    
                processed_samples.extend(compute_gae_advantages(player_samples, rewards, terminal_reward))

            entropy_accum = 0.0
            entropy_count = 0
            for sample_obj, _ in samples[0]:
                if sample_obj is not None and hasattr(sample_obj, 'policy') and len(sample_obj.policy) > 0:
                    policy_probs = [max(1e-8, p) for p in sample_obj.policy if p > 0]
                    p_sum = sum(policy_probs)
                    if p_sum > 0:
                        import math
                        entropy = -sum((p/p_sum) * math.log(p/p_sum) for p in policy_probs)
                        entropy_accum += entropy
                        entropy_count += 1
                        
            result_queue.put(("PLAY_SELF_COMPLETE", (worker_id, processed_samples, result, final_turn, rc_worker, entropy_accum, entropy_count, action_counts, played_cards, expert_log)))
            
        elif cmd == "EVAL":
            sample_deck, opponent_deck, opponent_name = args
            
            try:
                obs, start_data = battle_start(sample_deck, opponent_deck)
                if start_data.errorPlayer >= 0:
                    result_queue.put(("EVAL_COMPLETE", (worker_id, opponent_name, -1.0)))
                    continue
            except Exception as e:
                result_queue.put(("EVAL_COMPLETE", (worker_id, opponent_name, -1.0)))
                continue
                
            your_index = obs["current"]["yourIndex"]
            step_count = 0
            while True:
                step_count += 1
                if obs["current"]["result"] >= 0 or step_count > 400:
                    break
                
                curr_player = obs["current"]["yourIndex"]
                
                if curr_player == your_index:
                    turn = obs["current"]["turn"] if obs.get("current") else 1
                    if turn <= 1:
                        eval_search_count = 35
                    elif turn <= 3:
                        eval_search_count = 20
                    elif turn <= 8:
                        eval_search_count = 15
                    else:
                        eval_search_count = 10
                        
                    selected, _ = mcts_agent(obs, sample_deck, client, search_count=eval_search_count)
                else:
                    if opponent_name.startswith("Rulebasedmodel"):
                        try:
                            selected = rule_based_opponent_agent(opponent_name, obs)
                        except Exception as e:
                            selected = random_agent(obs)
                    else:
                        selected = random_agent(obs)
                try:
                    obs = battle_select(selected)
                except IndexError:
                    selected = random_agent(obs)
                    obs = battle_select(selected)
                
            battle_finish()
            result = obs["current"]["result"]
            
            if result == 2:
                outcome = 0.5
            elif result == your_index:
                outcome = 1.0
            else:
                outcome = 0.0
                
            result_queue.put(("EVAL_COMPLETE", (worker_id, opponent_name, outcome)))
