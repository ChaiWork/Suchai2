import json, os

files = ["88829519.json", "88830073 (1).json", "88830610.json"]

for fname in files:
    data = json.load(open(fname, "r", encoding="utf-8"))
    steps = data.get("steps", [])
    info = data.get("info", {})
    rewards = data.get("rewards", [])
    team_names = info.get("TeamNames", ["", ""])
    suchai_idx = 0 if team_names[0] == "Suchai" else 1
    opp_name = team_names[1 - suchai_idx]

    print("="*80)
    print(f"REPLAY: {fname} | Episode: {info.get('EpisodeId')} | Agent: Suchai (Player {suchai_idx}) vs {opp_name} | Rewards: {rewards}")
    print("="*80)

    for i, s in enumerate(steps):
        s_data = s[suchai_idx]
        obs = s_data.get("observation")
        if not isinstance(obs, dict):
            continue
        curr = obs.get("current")
        if not curr:
            continue

        turn = curr.get("turn", 0)
        p_me = curr["players"][suchai_idx]
        p_opp = curr["players"][1 - suchai_idx]

        hand = [c.get("name") for c in p_me.get("hand", []) if isinstance(c, dict)]
        act = [c.get("name") for c in p_me.get("active", []) if isinstance(c, dict)]
        bench = [c.get("name") for c in p_me.get("bench", []) if isinstance(c, dict)]

        opp_act = [c.get("name") for c in p_opp.get("active", []) if isinstance(c, dict)]
        opp_bench = [c.get("name") for c in p_opp.get("bench", []) if isinstance(c, dict)]

        sel = obs.get("select", {})
        ctx = sel.get("context", "")
        sel_type = sel.get("type", "")
        opts = sel.get("option", [])

        action = s_data.get("action")

        opt_strs = []
        if isinstance(opts, list):
            for opt in opts:
                if isinstance(opt, dict):
                    opt_strs.append(f"{opt.get('type')}:{opt.get('name', opt.get('id', ''))}")
                else:
                    opt_strs.append(str(opt))

        if opts and (sel_type in ("Main", "Card", "YesNo") or "Search" in str(ctx) or "Select" in str(ctx)):
            print(f"Step {i:03d} | Turn {turn:02d} | Ctx: {ctx} ({sel_type}) | Hand({len(hand)}): {hand} | Act: {act} | Bench: {bench} | OppAct: {opp_act} | Opts: {opt_strs[:6]} | Selected: {action}")
