import json
import glob
import os

iono3_dir = "iono3"
json_files = glob.glob(os.path.join(iono3_dir, "*.json"))

print(f"Analyzing {len(json_files)} replay JSONs in {iono3_dir}/\n")

reports = []

for filepath in sorted(json_files):
    fname = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    steps = data if isinstance(data, list) else data.get("steps", [])
    
    # Identify agent index (Mewtwo ex deck)
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
    mewtwo_attacks = 0
    spidops_attacks = 0
    benched_spidops_energy_attached = 0
    mewtwo_erasure_ball_ko_discards = 0
    
    bench_energies_history = []
    
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
            
            act_me = p_me.get("active", [])
            act_pk = act_me[0] if (act_me and isinstance(act_me[0], dict)) else {}
            act_id = act_pk.get("id", -1)
            
            bench_me = p_me.get("bench", [])
            b_energies = []
            spidops_b_energy = 0
            if isinstance(bench_me, list):
                for b in bench_me:
                    if isinstance(b, dict):
                        b_id = b.get("id", -1)
                        b_en = len(b.get("energyCards", [])) if isinstance(b.get("energyCards"), list) else 0
                        b_energies.append(b_en)
                        if b_id in (401, 400):  # Spidops / Tarountula
                            spidops_b_energy += b_en
                            
            bench_energies_history.append(sum(b_energies))
            
            action = s.get("action")
            select = s.get("select", {})
            options = select.get("option", [])
            
            if isinstance(action, list) and len(action) > 0 and action[0] is not None and isinstance(options, list):
                sel_idx = action[0] if isinstance(action[0], int) else (action[0][0] if isinstance(action[0], list) and len(action[0]) > 0 else -1)
                if isinstance(sel_idx, int) and 0 <= sel_idx < len(options):
                    opt = options[sel_idx]
                    if isinstance(opt, dict):
                        opt_type = opt.get("type")
                        if opt_type == 13:  # ATTACK
                            if act_id == 431:
                                mewtwo_attacks += 1
                            elif act_id in (401, 400):
                                spidops_attacks += 1
                        elif opt_type == 8:  # ATTACH ENERGY
                            target_id = opt.get("targetId", -1)
                            if target_id in (401, 400):
                                benched_spidops_energy_attached += 1

    reports.append({
        "file": fname,
        "actions": len(steps),
        "turns": last_turn,
        "mewtwo_attacks": mewtwo_attacks,
        "spidops_attacks": spidops_attacks,
        "avg_bench_energy": sum(bench_energies_history) / max(1, len(bench_energies_history)),
        "max_bench_energy": max(bench_energies_history) if bench_energies_history else 0
    })

print("="*105)
print(f"{'FILE':<20} | {'ACTIONS':<8} | {'TURNS':<6} | {'MEWTWO ATK':<11} | {'SPIDOPS ATK':<12} | {'AVG BENCH EN':<14} | {'MAX BENCH EN'}")
print("="*105)

for r in reports:
    print(f"{r['file']:<20} | {r['actions']:<8} | {r['turns']:<6} | {r['mewtwo_attacks']:<11} | {r['spidops_attacks']:<12} | {r['avg_bench_energy']:<14.2f} | {r['max_bench_energy']}")

print("="*105)
