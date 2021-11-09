import os
import sys

base_path = os.path.dirname(__file__)
modist_path = os.path.join(base_path, "../src")
modist_path = os.path.realpath(modist_path)
if modist_path not in sys.path:
    sys.path.append(modist_path)

import modist