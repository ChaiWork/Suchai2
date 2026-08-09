import json

filepath = "iono3/match_04_win.json"

with open(filepath, "r", encoding="utf-8") as f:
    data = json.load(f)

steps = data if isinstance(data, list) else data.get("steps", [])

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
print(f"TRACING FILE: {filepath} | Total Steps: {len(steps)} | Agent Index: {agent_idx}\n")

prev_turn = -1
prev_prizes_me = -1
prev_prizes_opp = -1

for i, s in enumerate(steps):
    if not isinstance(s, dict):
        continue
    curr = s.get("current")
    if not isinstance(curr, dict):
        continue
        
    turn = curr.get("turn", 0)
    p_me = curr["players"][agent_idx]
    p_opp = curr["players"][opp_idx]
    
    prizes_me = len(p_me.get("prize", []))
    prizes_opp = len(p_opp.get("prize", []))
    
    act_me = [f"{c.get('name', c.get('id'))}({len(c.get('energyCards', []))}en, {c.get('hp')}/{c.get('maxHp')}hp)" for c in p_me.get("active", []) if isinstance(c, dict)]
    bench_me = [f"{c.get('name', c.get('id'))}({len(c.get('energyCards', []))}en)" for c in p_me.get("bench", []) if isinstance(c, dict)]
    
    act_opp = [f"{c.get('name', c.get('id'))}({c.get('hp')}/{c.get('maxHp')}hp)" for c in p_opp.get("active", []) if isinstance(c, dict)]
    bench_opp = [f"{c.get('name', c.get('id'))}" for c in p_opp.get("bench", []) if isinstance(c, dict)]
    
    action = s.get("action")
    select = s.get("select", {})
    options = select.get("option", [])
    
    logs = s.get("logs", [])
    log_types = [l.get("type") for l in logs if isinstance(l, dict)]
    
    # Print key milestones (turn change, prize change, attack, or every 10 steps)
    if turn != prev_turn or prizes_me != prev_prizes_me or prizes_opp != prev_prizes_opp or "Attack" in log_types or "HpChange" in log_types:
        opt_strs = [f"{o.get('type')}:{o.get('name', o.get('id', ''))}" for o in options if isinstance(o, dict)][:3]
        print(f"Step {i:03d} | Turn {turn:02d} | Prizes Us:{prizes_me}/Opp:{prizes_opp} | ActUs: {act_me} | BenchUs({len(bench_me)}): {bench_me} | ActOpp: {act_opp} | BenchOpp({len(bench_opp)}): {bench_opp[:2]} | Action: {action} | Logs: {log_types}")
        
    prev_turn = turn
    prev_prizes_me = prizes_me
    prev_prizes_opp = prizes_opp
