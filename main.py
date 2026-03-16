import numpy as np
from collections import defaultdict
from itertools import combinations

class Fast3SAT:
  def __init__(self, num_vars, clauses):
    self.n = num_vars
    self.m = len(clauses)
    self.clauses = clauses
    self.ver_index = {f'X{i}': i-1 for i in range(1, num_vars + 1)}
