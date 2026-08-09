import json
import glob
import os
from collections import Counter

lost_dir = "LOST"
json_files = glob.glob(os.path.join(lost_dir, "*.json"))

print(f"Analyzing {len(json_files)} lost match replays in detail...\n")

results_summary = []

for filepath in json_files:
    filename = os.path.basename(filepath)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    info = data.get("info", {})
    rewards = data.get("rewards", [0, 0])
    team_names = info.get("TeamNames", ["Player 0", "Player 1"])
    
    steps = data.get("steps", [])
    
    # Identify agent index (our agent uses Team Rocket Mewtwo ex deck: 431, 400, 401, 414, 434)
    agent_idx = 0
    opp_idx = 1
    
    # Find which player has Mewtwo ex deck cards
    for s in steps[:5]:
        if not isinstance(s, list):
            continue
        for p_i in range(len(s)):
            obs_i = s[p_i].get("observation", {})
            if isinstance(obs_i, dict):
                curr_i = obs_i.get("current")
                if isinstance(curr_i, dict):
                    players_i = curr_i.get("players", [])
                    if isinstance(players_i, (list, tuple)) and len(players_i) > p_i:
                        h = [c.get("id", -1) for c in players_i[p_i].get("hand", []) if isinstance(c, dict)]
                        if any(cid in (431, 400, 401, 414, 434) for cid in h):
                            agent_idx = p_i
                            opp_idx = 1 - agent_idx
                            break

    my_reward = rewards[agent_idx] if len(rewards) > agent_idx else -1
    opp_reward = rewards[opp_idx] if len(rewards) > opp_idx else -1

    last_turn = 0
    final_prizes_me = 6
    final_prizes_opp = 6
    final_deck_me = 60
    final_deck_opp = 60
    final_active_me = "None"
    final_active_opp = "None"
    final_bench_me = []
    final_bench_opp = []
    
    opp_cards_seen = set()
    battle_cage_in_play = False
    tr_factory_in_play = False
    
    # Trace steps
    for i, step in enumerate(steps):
        if not isinstance(step, list) or len(step) <= agent_idx:
            continue
        s_data = step[agent_idx]
        obs = s_data.get("observation", {})
        if not isinstance(obs, dict):
            continue
        curr = obs.get("current", {})
        if not curr:
            continue
        
        turn = curr.get("turn", 0)
        last_turn = max(last_turn, turn)
        
        players = curr.get("players", [])
        if len(players) > 1:
            p_me = players[agent_idx]
            p_opp = players[opp_idx]
            
            final_prizes_me = len(p_me.get("prize", []))
            final_prizes_opp = len(p_opp.get("prize", []))
            
            final_deck_me = p_me.get("deckCount", 60)
            final_deck_opp = p_opp.get("deckCount", 60)
            
            act_me = p_me.get("active", [])
            if act_me and isinstance(act_me[0], dict):
                final_active_me = act_me[0].get("name", str(act_me[0].get("id", -1)))
                
            act_opp = p_opp.get("active", [])
            if act_opp and isinstance(act_opp[0], dict):
                final_active_opp = act_opp[0].get("name", str(act_opp[0].get("id", -1)))
                opp_cards_seen.add(final_active_opp)
                
            bench_me = [c.get("name", str(c.get("id", -1))) for c in p_me.get("bench", []) if isinstance(c, dict)]
            bench_opp = [c.get("name", str(c.get("id", -1))) for c in p_opp.get("bench", []) if isinstance(c, dict)]
            final_bench_me = bench_me
            final_bench_opp = bench_opp
            for b_c in bench_opp:
                opp_cards_seen.add(b_c)
                
            stadium = curr.get("stadium", [])
            if stadium and isinstance(stadium[0], dict):
                st_id = stadium[0].get("id", -1)
                if st_id == 1264:
                    battle_cage_in_play = True
                elif st_id == 1257:
                    tr_factory_in_play = True

    # Classify opponent deck archetype
    opp_archetype = "Unknown"
    opp_cards_str = " ".join(opp_cards_seen).lower()
    if "dragapult" in opp_cards_str or "drakloak" in opp_cards_str or "dreepy" in opp_cards_str:
        opp_archetype = "Dragapult ex"
    elif "clefairy" in opp_cards_str or "togekiss" in opp_cards_str or "togepi" in opp_cards_str:
        opp_archetype = "Lillie's Clefairy ex"
    elif "mewtwo" in opp_cards_str:
        opp_archetype = "Mewtwo ex Mirror"
    elif "alakazam" in opp_cards_str or "kadabra" in opp_cards_str or "abra" in opp_cards_str:
        opp_archetype = "Alakazam"
    elif "grimmsnarl" in opp_cards_str:
        opp_archetype = "Grimmsnarl ex"
    elif "hydrapple" in opp_cards_str:
        opp_archetype = "Hydrapple ex"
    elif "garchomp" in opp_cards_str:
        opp_archetype = "Garchomp ex"
    else:
        opp_archetype = f"Other ({', '.join(list(opp_cards_seen)[:2])})"

    # Determine failure mode
    if my_reward == -1 and opp_reward == -1:
        loss_reason = "Error/Invalid Move Timeout"
    elif final_deck_me <= 0:
        loss_reason = "Deckout (0 cards remaining)"
    elif len(final_bench_me) == 0 and final_prizes_me > 0:
        loss_reason = "Bench Out (No benched Pokemon)"
    elif final_prizes_me == 6 and final_prizes_opp < 6:
        loss_reason = "Donk/Early KO (0 prizes taken by us)"
    elif final_prizes_me > final_prizes_opp:
        loss_reason = f"Prize Deficit (Prizes left: Us {final_prizes_me} vs Opp {final_prizes_opp})"
    elif final_prizes_me == final_prizes_opp:
        loss_reason = f"Tie/Time Limit (Prizes left: {final_prizes_me})"
    else:
        loss_reason = f"Defeat (Prizes left: Us {final_prizes_me} vs Opp {final_prizes_opp})"

    opp_name = str(team_names[opp_idx]).encode('ascii', 'ignore').decode('ascii')
    if not opp_name:
        opp_name = f"Player_{opp_idx}"

    results_summary.append({
        "file": filename,
        "opponent_name": opp_name,
        "opp_archetype": opp_archetype,
        "turns": last_turn,
        "my_reward": my_reward,
        "opp_reward": opp_reward,
        "prizes_us": final_prizes_me,
        "prizes_opp": final_prizes_opp,
        "deck_us": final_deck_me,
        "loss_reason": loss_reason,
        "active_us": final_active_me,
        "active_opp": final_active_opp,
        "bench_us_cnt": len(final_bench_me),
        "battle_cage_played": battle_cage_in_play,
        "tr_factory_played": tr_factory_in_play
    })

print("="*115)
print(f"{'FILE':<18} | {'OPPONENT ARCHETYPE':<22} | {'TURNS':<5} | {'PRIZES(US/OPP)':<15} | {'DECK':<5} | {'LOSS REASON'}")
print("="*115)

for s in results_summary:
    prizes_str = f"{s['prizes_us']} / {s['prizes_opp']}"
    print(f"{s['file']:<18} | {s['opp_archetype']:<22} | {s['turns']:<5} | {prizes_str:<15} | {s['deck_us']:<5} | {s['loss_reason']}")

print("="*115)

print("\n--- LOSS REASON BREAKDOWN ---")
reasons = Counter([s["loss_reason"] for s in results_summary])
for r, c in reasons.most_common():
    print(f"  * {r}: {c} matches ({c/len(results_summary)*100:.1f}%)")

print("\n--- OPPONENT DECK ARCHETYPE BREAKDOWN ---")
archs = Counter([s["opp_archetype"] for s in results_summary])
for a, c in archs.most_common():
    print(f"  * {a}: {c} matches ({c/len(results_summary)*100:.1f}%)")

print("\n--- STADIUM USAGE IN LOSSES ---")
cage_count = sum(1 for s in results_summary if s["battle_cage_played"])
factory_count = sum(1 for s in results_summary if s["tr_factory_played"])
print(f"  * Battle Cage in play at match end: {cage_count} / {len(results_summary)} ({cage_count/len(results_summary)*100:.1f}%)")
print(f"  * TR Factory in play at match end:  {factory_count} / {len(results_summary)} ({factory_count/len(results_summary)*100:.1f}%)")

