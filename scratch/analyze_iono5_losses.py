import json
import glob
import os

iono5_dir = "iono5"
json_files = glob.glob(os.path.join(iono5_dir, "*.json"))

print(f"Parsing {len(json_files)} replay JSONs in {iono5_dir}/\n")

reports = []

for filepath in sorted(json_files):
    fname = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    steps = data if isinstance(data, list) else data.get("steps", [])
    info = data.get("info", {}) if isinstance(data, dict) else {}
    
    agent_idx = 0
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
    final_bench_me = []
    
    bench_energies_history = []
    tr_count_history = []
    
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
            
            act_me = p_me.get("active", [])
            bench_me = p_me.get("bench", [])
            final_bench_me = bench_me
            
            b_energies = [len(b.get("energyCards", [])) for b in bench_me if isinstance(b, dict)]
            bench_energies_history.append(sum(b_energies))
            
            TR_IDS = {400, 401, 414, 431, 434}
            all_my_ids = [act_me[0].get("id", -1)] if (act_me and isinstance(act_me[0], dict)) else []
            all_my_ids.extend([c.get("id", -1) for c in bench_me if isinstance(c, dict)])
            tr_cnt = sum(1 for cid in all_my_ids if cid in TR_IDS)
            tr_count_history.append(tr_cnt)

    is_win = "win" in fname.lower()
    
    loss_reason = "WIN" if is_win else "Unknown"
    if not is_win:
        if final_deck_me <= 0:
            loss_reason = "Deckout"
        elif len(final_bench_me) == 0:
            loss_reason = "Bench Out (0 Benched Pokemon)"
        elif final_prizes_me == 6:
            loss_reason = "Donk (0 Prizes Taken)"
        else:
            loss_reason = f"Defeat (Us:{final_prizes_me} vs Opp:{final_prizes_opp})"

    reports.append({
        "file": fname,
        "is_win": is_win,
        "turns": last_turn,
        "prizes_us": final_prizes_me,
        "prizes_opp": final_prizes_opp,
        "deck_us": final_deck_me,
        "bench_cnt": len(final_bench_me),
        "avg_bench_en": sum(bench_energies_history) / max(1, len(bench_energies_history)),
        "max_bench_en": max(bench_energies_history) if bench_energies_history else 0,
        "avg_tr_cnt": sum(tr_count_history) / max(1, len(tr_count_history)),
        "loss_reason": loss_reason
    })

print("="*115)
print(f"{'FILE':<18} | {'RESULT':<6} | {'TURNS':<5} | {'PRIZES(US/OPP)':<15} | {'BENCH':<5} | {'AVG BENCH EN':<14} | {'AVG TR':<7} | {'LOSS REASON'}")
print("="*115)

for r in reports:
    prizes_str = f"{r['prizes_us']} / {r['prizes_opp']}"
    res_str = "WIN" if r['is_win'] else "LOSS"
    print(f"{r['file']:<18} | {res_str:<6} | {r['turns']:<5} | {prizes_str:<15} | {r['bench_cnt']:<5} | {r['avg_bench_en']:<14.2f} | {r['avg_tr_cnt']:<7.2f} | {r['loss_reason']}")

print("="*115)
