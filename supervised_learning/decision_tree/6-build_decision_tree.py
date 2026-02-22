#!/usr/bin/env python3
"""
Decision Tree Classifier
"""


import numpy as np


class Node:
    """Node class for decision tree"""
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

    def max_depth_below(self):
        """Returns the maximum depth of the tree below this node"""
        left_child_depth = self.left_child.max_depth_below()
        right_child_depth = self.right_child.max_depth_below()
        return max(left_child_depth, right_child_depth)

    def count_nodes_below(self, only_leaves=False):
        """Returns the number of nodes below this node"""
        left_nodes_count = self.left_child.count_nodes_below(
            only_leaves=only_leaves
        )
        right_nodes_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves
        )
        if only_leaves:
            return left_nodes_count + right_nodes_count
        return 1 + left_nodes_count + right_nodes_count

    def __str__(self):
        """Returns a string representation of the node"""
        if self.is_root:
            node_str = f"""root [feature={self.feature},
                threshold={self.threshold}]"""
        else:
            node_str = f"""-> node [feature={self.feature},
                threshold={self.threshold}]"""

        left_str = str(self.left_child)
        right_str = str(self.right_child)

        left_with_prefix = self.left_child_add_prefix(left_str)
        right_with_prefix = self.right_child_add_prefix(right_str)

        result = node_str
        if left_with_prefix:
            result += "\n" + left_with_prefix
        if right_with_prefix:
            result += right_with_prefix
        return result

    def left_child_add_prefix(self, text):
        """Adds prefix to the left child text"""
        lines = text.split("\n")
        new_text = "    +--"+lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  "+x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Adds prefix to the right child text"""
        lines = text.split('\n')
        new_text = "    +--" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    " + line + "\n"
        return new_text.rstrip('\n')

    def get_leaves_below(self):
        """Returns the leaves below this node"""
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Updates the bounds of the node"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()
            if child == self.left_child:
                child.lower[self.feature] = self.threshold
            else:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Updates the indicator function for the node"""

        def is_large_enough(x):
            """Returns a 1D numpy array of size `n_individuals`"""
            return np.all(np.array([x[:, key] > self.lower[key]
                                    for key in self.lower.keys()]), axis=0)

        def is_small_enough(x):
            """Returns a 1D numpy array of size `n_individuals`"""
            return np.all(np.array([x[:, key] <= self.upper[key]
                                    for key in self.upper.keys()]), axis=0)

        self.indicator = lambda x: \
            np.all(np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Returns the prediction for a single instance"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Leaf class for decision tree"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Returns the depth of the leaf node"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Returns the number of nodes below this leaf"""
        return 1

    def __str__(self):
        """Returns a string representation of the leaf"""
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """Returns the leaves below this leaf"""
        return [self]

    def update_bounds_below(self):
        """Updates the bounds of the leaf node"""
        pass

    def pred(self, x):
        """Returns the prediction for a single instance"""
        return self.value


class Decision_Tree():
    """Decision Tree class"""
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
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

    def depth(self):
        """Returns the depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Returns the number of nodes in the tree"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Returns a string representation of the tree"""
        return self.root.__str__()

    def get_leaves(self):
        """Returns the leaves of the tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Updates the bounds of the tree"""
        self.root.update_bounds_below()

    def pred(self, x):
        """Returns the prediction for a single instance"""
        return self.root.pred(x)

    def update_predict(self):
        """Updates the prediction function for the tree"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            """Returns the predictions for an array of instances"""
            predictions = np.zeros(A.shape[0], dtype=int)
            for i, x in enumerate(A):
                for leaf in leaves:
                    if leaf.indicator(np.array([x])):
                        predictions[i] = leaf.value
                        break
            return predictions
        self.predict = predict
