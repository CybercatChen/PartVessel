from __future__ import absolute_import
import collections
import torch
import numpy as np
from functools import reduce


def encode_structure_fold(fold, root):
    def encode_node(node):
        if node is None:
            return
        if node.is_leaf():
            return fold.add('leafEncoder', node.radius)
        else:
            left = encode_node(node.left)
            right = encode_node(node.right)
            if left is not None and right is not None:
                return fold.add('bifurcationEncoder', node.radius, right, left)
            elif right is not None:
                return fold.add('internalEncoder', node.radius, right)
            elif left is not None:
                return fold.add('internalEncoder', node.radius, left)

    encoding = encode_node(root)
    return fold.add('sampleEncoder', encoding)


def decode_structure_fold(fold, encoded_vector, root):
    def calculate_base_loss(fold, vector, node, radius):
        multiple = np.round((node.maxlevel + 1 - node.level) / node.treelevel, decimals=2)
        label = fold.add('nodeClassifier', vector)
        loss_ce = fold.add('classifyLossEstimator', label, node)
        loss_att = fold.add('MSE_loss', node, radius)
        loss_ce_mult = fold.add('Mult', multiple, loss_ce)
        return fold.add('Add', loss_ce_mult, loss_att)

    def decode_leaf(fold, vector, node):
        radius = fold.add('featureDecoder', vector)
        return calculate_base_loss(fold, vector, node, radius)

    def decode_internal(fold, vector, node):
        right, radius = fold.add('internalDecoder', vector).split(2)
        child_node = node.right if node.right else node.left
        child_loss = decode_node(fold, right, child_node)
        base_loss = calculate_base_loss(fold, vector, node, radius)
        return fold.add('Add', base_loss, child_loss)

    def decode_bifurcation(fold, vector, node):
        left, right, radius = fold.add('bifurcationDecoder', vector).split(3)
        child_losses = [decode_node(fold, v, n) for v, n in [(left, node.left), (right, node.right)] if n is not None]
        base_loss = calculate_base_loss(fold, vector, node, radius)
        return reduce(lambda x, y: fold.add('Add', x, y), [base_loss] + child_losses)

    def decode_node(fold, vector, node):
        decoders = {
            0: decode_leaf,
            1: decode_internal,
            2: decode_bifurcation
        }
        return decoders[node.child_num()](fold, vector, node)

    decoded_vector = fold.add('sampleDecoder', encoded_vector)
    return decode_node(fold, decoded_vector, root)


class Fold(object):
    class Node(object):
        def __init__(self, op, step, index, *args):
            self.op = op
            self.step = step
            self.index = index
            self.args = args
            self.split_idx = -1
            self.batch = True

        def split(self, num):
            nodes = []
            for idx in range(num):
                nodes.append(Fold.Node(self.op, self.step, self.index, *self.args))
                nodes[-1].split_idx = idx
            return tuple(nodes)

        def get(self, values):
            if self.split_idx >= 0:
                return values[self.step][self.op][self.split_idx][self.index]
            else:
                return values[self.step][self.op][self.index]

    def __init__(self, volatile=False, cuda=False, variable=True):
        self.steps = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.cached_nodes = collections.defaultdict(dict)
        self.total_nodes = 0
        self.volatile = volatile
        self.cuda = cuda
        self.variable = variable

    def add(self, op, *args):
        self.total_nodes += 1

        if args not in self.cached_nodes[op]:
            step = max([0] + [arg.step + 1 for arg in args if isinstance(arg, Fold.Node)])
            node = Fold.Node(op, step, len(self.steps[step][op]), *args)
            self.steps[step][op].append(args)
            self.cached_nodes[op][args] = node

        return self.cached_nodes[op][args]

    def _batch_args(self, arg_lists, values, op):
        def process_arg(arg):
            if isinstance(arg[0], Fold.Node) and arg[0].batch:
                return torch.stack([x.get(values) for x in arg])
            elif isinstance(arg[0], torch.Tensor):
                return torch.stack(arg)
            else:
                return self._process_special_arg(arg, op)

        return [process_arg(arg) for arg in arg_lists]

    def _process_special_arg(self, arg, op):
        special_ops = {
            "classifyLossEstimator": lambda a: [x.child_num() for x in a],
            "MSE_loss": lambda a: [x.radius for x in a],
            "Mult": lambda a: a if isinstance(a, torch.Tensor) else list(a),
            "sampleEncoder": lambda a: a
        }
        if op in special_ops:
            return special_ops[op](arg)
        return arg[0].radius

    def apply(self, nn, nodes):
        values = {}
        for step in sorted(self.steps.keys()):
            values[step] = {}
            for op in self.steps[step]:
                func = getattr(nn, op)
                batched_args = self._batch_args(zip(*self.steps[step][op]), values, op)
                res = func(*batched_args)

                if isinstance(res, (list, tuple)):
                    values[step][op] = list(res)
                elif len(res.shape) == 1 and op not in ['Add', 'Mult']:
                    values[step][op] = res.unsqueeze(0)
                else:
                    values[step][op] = res

        try:
            return self._batch_args(nodes, values, op)
        except Exception as e:
            print(f"Cannot batch: {e}")
            raise
