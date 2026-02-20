#!/usr/bin/env python3
"""
Decision Tree implementation with __str__ methods
"""

import numpy as np

class Node:
    """Node in a decision tree"""
    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None,
                 is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def left_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text

    def __str__(self):
        text = f"node [feature={self.feature}, threshold={self.threshold}]"
        if self.left_child:
            text = self.left_child_add_prefix(str(self.left_child)) + text if self.depth != 0 else text
        if self.right_child:
            text += "\n" + self.right_child_add_prefix(str(self.right_child))
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]\n" + \
                   (self.left_child_add_prefix(str(self.left_child)) if self.left_child else "") + \
                   (self.right_child_add_prefix(str(self.right_child)) if self.right_child else "")
        return text.strip()

class Leaf(Node):
    """Leaf node in a decision tree"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        return f"-> leaf [value={self.value}]"

class Decision_Tree:
    """Decision tree class"""
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def __str__(self):
        return self.root.__str__()
