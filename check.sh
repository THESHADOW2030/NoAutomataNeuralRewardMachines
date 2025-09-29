python3 - <<'PY'
import sys, traceback

imports = [
    ("torch", "import torch"),
    ("torch.nn", "import torch.nn as nn"),
    ("torch.nn.functional", "import torch.nn.functional as F"),
    ("dot2pythomata", "from RL.NRM.utils import dot2pythomata"),
    ("transacc2pythomata", "from RL.NRM.utils import transacc2pythomata"),
    ("MinimizableMooreMachine", "from RL.NRM.Minimization import MinimizableMooreMachine"),
]

for name, stmt in imports:
    try:
        exec(stmt, {})
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=1)
print('\\nsys.path:')
import pprint
pprint.pprint(sys.path)
PY