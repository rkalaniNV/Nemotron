import os, sys
_root = os.path.dirname(__file__)
for p in (os.path.join(_root, "src"), _root):
    if p not in sys.path:
        sys.path.insert(0, p)
