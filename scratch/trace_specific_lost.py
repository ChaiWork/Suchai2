import json

def trace_file(filepath):
    print("="*90)
    print(f"TRACING FILE: {filepath}")
    print("="*90)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    steps = data.get("steps", [])
    agent_idx = 0
    # Determine agent index
    for s in steps[:5]:
        if isinstance(s, list) and len(s) > 1:
            for p_i in (0, 1):
                obs = s[p_i].get("observation", {})
                if isinstance(obs, dict) and isinstance(obs.get("current"), dict):
                    players = obs["current"].get("players", [])
                    if isinstance(players, (list, tuple)) and len(players) > p_i:
                        h = [c.get("id", -1) for c in players[p_i].get("hand", []) if isinstance(c, dict)]
                        if any(cid in (431, 400, 401, 414, 434) for cid in h):
                            agent_idx = p_i
                            break
                            
    print(f"Detected Agent Index: {agent_idx}\n")
    
    for i, step in enumerate(steps):
        if not isinstance(step, list) or len(step) <= agent_idx:
            continue
        s_data = step[agent_idx]
        action = s_data.get("action")
        obs = s_data.get("observation", {})
        if not isinstance(obs, dict) or not isinstance(obs.get("current"), dict):
            continue
        curr = obs["current"]
        turn = curr.get("turn", 0)
        p_me = curr["players"][agent_idx]
        p_opp = curr["players"][1 - agent_idx]
        
        act_me = [c.get("name", str(c.get("id"))) for c in p_me.get("active", []) if isinstance(c, dict)]
        bench_me = [c.get("name", str(c.get("id"))) for c in p_me.get("bench", []) if isinstance(c, dict)]
        prizes_me = len(p_me.get("prize", []))
        prizes_opp = len(p_opp.get("prize", []))
        
        opts = obs.get("select", {}).get("option", [])
        opt_summary = [f"{o.get('type')}:{o.get('name', o.get('id', ''))}" for o in opts if isinstance(o, dict)][:5]
        
        print(f"Step {i:02d} | Turn {turn:02d} | Prizes Us:{prizes_me}/Opp:{prizes_opp} | Act: {act_me} | Bench({len(bench_me)}): {bench_me} | Action: {action} | Opts: {opt_summary}")

trace_file("LOST/91228405.json")
trace_file("LOST/91237383.json")
