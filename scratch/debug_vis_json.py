import json

with open("iono/match_01_win.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Top level type: {type(data)}")
print(f"Length: {len(data)}")

step0 = data[0]
print("Step 0 keys:", step0.keys() if isinstance(step0, dict) else "list len " + str(len(step0)))

if isinstance(step0, list):
    print("Step 0 element 0 keys:", step0[0].keys() if isinstance(step0[0], dict) else type(step0[0]))
    if len(step0) > 1:
        print("Step 0 element 1 keys:", step0[1].keys() if isinstance(step0[1], dict) else type(step0[1]))

for i in list(range(18, 25)) + list(range(64, 72)):
    if i < len(data):
        s = data[i]
        curr = s.get("current", {})
        turn = curr.get("turn", 0)
        action = s.get("action")
        selected = s.get("selected")
        logs = s.get("logs", [])
        select = s.get("select", {})
        opts = select.get("option", [])
        opt_strs = [f"{o.get('type')}:{o.get('name', o.get('id', ''))}" for o in opts if isinstance(o, dict)]
        print(f"--- STEP {i:03d} (Turn {turn:02d}) ---")
        print(f"Action: {action} | Selected: {selected}")
        print(f"Options: {opt_strs}")
        print(f"Logs: {logs[:3]}")
