import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))

from codetf.models import model_zoo
print(model_zoo)