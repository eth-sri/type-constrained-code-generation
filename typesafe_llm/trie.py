"""
MIT License

Copyright (c) 2022 Kanishk Gandhi
Copyright (c) 2024 Anonymized (adapted)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#!/usr/bin/env python3

import collections
from typing import Iterable

from typesafe_llm.util import pflush


# Trie representation of a vocabulary.
class Trie:
    def __init__(self, value=None, enforce_token_maximality=True):
        self._children = collections.defaultdict(
            lambda: Trie(enforce_token_maximality=enforce_token_maximality)
        )
        self._value = [value] if value is not None else []
        self._enforce_token_maximality = enforce_token_maximality

    def insert(self, key, value, depth=0):
        if len(key) == depth:
            self._value.append(value)
        else:
            self._children[key[depth]].insert(key, value, depth + 1)

    @staticmethod
    def from_vocabulary(vocab: Iterable[str], enforce_token_maximality: bool = True):
        t = Trie(enforce_token_maximality=enforce_token_maximality)

        for i, token in enumerate(vocab):
            if token:
                t.insert(token, i)

        return t

    def antimonotonic_filter(
        self, parse_fn, states, key="", _pflush=pflush
    ) -> list[tuple[str, list]]:
        # NOTE only works when all keys are unique!
        # key_d = json.dumps(key)[1:-1]
        # _pflush(key_d)
        this_node_valid = parse_fn(states, key) if key else states

        if not this_node_valid:
            # Prune using anti-monotonicity: no children will be valid.
            # delete(key_d, _pflush)
            return []

        children_values = []

        for k, c in self._children.items():
            children_values += c.antimonotonic_filter(
                parse_fn, this_node_valid, k, _pflush=_pflush
            )

        this_value = [(v, this_node_valid) for v in self._value]
        # delete(key_d, _pflush)

        if self._enforce_token_maximality:
            # Only return maximal strings.
            if len(children_values) or not self._value:
                return children_values
            return this_value

        return this_value + children_values
