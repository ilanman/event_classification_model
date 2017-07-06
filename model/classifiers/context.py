"""
Provides context for importing sub package level modules
"""

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
