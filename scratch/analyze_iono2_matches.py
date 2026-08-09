import json
import glob
import os

iono2_dir = "iono2"
json_files = glob.glob(os.path.join(iono2_dir, "*.json"))

print(f"Parsing {len(json_files)} replay JSONs in {iono2_dir}/\n")

match_reports = []

for filepath in json_files:
    fname = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    steps = data if isinstance(data, list) else data.get("steps", [])
    
    agent_idx = 0
    # Determine agent player index (Mewtwo ex deck cards)
    for s in steps[:5]:
        if isinstance(s, dict) and "current" in s:
            curr = s["current"]
            players = curr.get("players", [])
            for p_i in (0, 1):
                if len(players) > p_i:
                    h = [c.get("id", -1) for c in players[p_i].get("hand", []) if isinstance(c, dict)]
                    if any(cid in (431, 400, 401, 414, 434) for cid in h):
                        agent_idx = p_i
                        break

    opp_idx = 1 - agent_idx
    
    last_turn = 0
    final_prizes_me = 6
    final_prizes_opp = 6
    final_deck_me = 60
    final_deck_opp = 60
    final_active_me = "None"
    final_active_opp = "None"
    final_bench_me = []
    final_bench_opp = []
    
    tr_count_history = []
    hand_size_history = []
    iono_disruptions_count = 0
    
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        curr = s.get("current")
        if not isinstance(curr, dict):
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
                
            bench_me = [c.get("name", str(c.get("id", -1))) for c in p_me.get("bench", []) if isinstance(c, dict)]
            bench_opp = [c.get("name", str(c.get("id", -1))) for c in p_opp.get("bench", []) if isinstance(c, dict)]
            final_bench_me = bench_me
            final_bench_opp = bench_opp
            
            # Track TR count & hand size
            TR_IDS = {400, 401, 414, 431, 434}
            all_my_ids = [act_me[0].get("id", -1)] if (act_me and isinstance(act_me[0], dict)) else []
            all_my_ids.extend([c.get("id", -1) for c in p_me.get("bench", []) if isinstance(c, dict)])
            tr_cnt = sum(1 for cid in all_my_ids if cid in TR_IDS)
            tr_count_history.append(tr_cnt)
            
            hand_me = p_me.get("hand", [])
            hand_size_history.append(len(hand_me))
            
            # Check logs for Iono played by opponent
            logs = s.get("logs", [])
            for lg in logs:
                if isinstance(lg, dict) and lg.get("type") == "Play" and lg.get("playerIndex") == opp_idx:
                    if lg.get("cardId") in (1218, 1259):  # Iono / disruption
                        iono_disruptions_count += 1

    is_win = "win" in fname.lower()
    
    match_reports.append({
        "file": fname,
        "is_win": is_win,
        "turns": last_turn,
        "prizes_us": final_prizes_me,
        "prizes_opp": final_prizes_opp,
        "deck_us": final_deck_me,
        "active_us": final_active_me,
        "active_opp": final_active_opp,
        "bench_us_cnt": len(final_bench_me),
        "avg_tr_count": sum(tr_count_history) / max(1, len(tr_count_history)),
        "min_tr_count": min(tr_count_history) if tr_count_history else 0,
        "avg_hand_size": sum(hand_size_history) / max(1, len(hand_size_history)),
        "iono_disruptions": iono_disruptions_count
    })

print("="*110)
print(f"{'FILE':<20} | {'RESULT':<6} | {'TURNS':<5} | {'PRIZES(US/OPP)':<15} | {'DECK':<5} | {'BENCH':<5} | {'AVG TR CNT':<10} | {'IONO DISRUPT'}")
print("="*110)

for m in sorted(match_reports, key=lambda x: x['file']):
    res_str = "WIN" if m['is_win'] else "LOSS"
    prizes_str = f"{m['prizes_us']} / {m['prizes_opp']}"
    print(f"{m['file']:<20} | {res_str:<6} | {m['turns']:<5} | {prizes_str:<15} | {m['deck_us']:<5} | {m['bench_us_cnt']:<5} | {m['avg_tr_count']:<10.2f} | {m['iono_disruptions']}")

print("="*110)
