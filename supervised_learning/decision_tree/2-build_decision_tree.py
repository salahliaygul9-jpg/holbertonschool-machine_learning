#!/usr/bin/env python3
"""
Defines Let's print our Tree
Classes:
    Node: no leaf in tree
    Leaf: leaf node and inherit from Node
    Decision_Tree: main clas
"""


class Node:
    """
    Structure of decision tree
    Attributes:
        feature: the property used for partition
        threshold: build nodes
        left_child, right_child: branches from nodes
        is_leaf: if the node is a leaf
        depth: depth of tree
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def left_child_add_prefix(self, text):
        """
         Add a prefix to the left child text
        """
        lines = text.strip().split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("    |  "+x)+"\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """
        Add a prefix to the right child text
        """
        lines = text.strip().split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("       "+x)+"\n"
        return (new_text)

    def __str__(self):
        """
        Present the nodes (non-leaf) in the tree
        Return:
            text: Node [feature=feature, threshold=threshold]
            concatenate between left_text and right_text
        """
        left_text = self.left_child.__str__() if self.left_child else ""
        right_text = self.right_child.__str__() if self.right_child else ""
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            text = f"-> node [feature={self.feature}, " \
                   f"threshold={self.threshold}]"
        if self.is_leaf:
            return f"    +---> leaf [value={self.value}]"
        else:
            left_text = self.left_child_add_prefix(left_text)
            right_text = self.right_child_add_prefix(right_text)
            return f"{text}\n{left_text}{right_text}"


class Leaf(Node):
    """
    A leaf node in a decision tree inheriting from class Node
    Attributes:
        value: the value for leaf node
        depth: the depth of the leaf node
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """Return the value of the leaf node"""
        return (f"-> leaf [value={self.value}]")


class Decision_Tree:
    """
    Decision tree of classification or regression
    Print tree
    """
    def __init__(self, root=None):
        """
        Initialize a decision tree with a root node
        """
        self.root = root if root else Node(is_root=True)

    def __str__(self):
        """Print tree like text starting from root node"""
        return self.root.__str__()
