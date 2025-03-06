import torch
from abc import abstractmethod


class Tnorm:
    def negation(self, a):
        return 1-a

    @staticmethod
    def get_tnorm(name):
        if name == 'product':
            return ProductTNorm()
        elif name == 'godel':
            return GodelTNorm()
        else:
            raise ValueError('Unknown t-norm: {}'.format(name))

    @abstractmethod
    def conjunction(self, a, b):
        pass

    def disjunction(self, a, b):
        return self.negation(
            self.conjunction(
                self.negation(a),
                self.negation(b)
            )
        )


class ProductTNorm(Tnorm):
    def conjunction(self, a, b):
        return a * b

class GodelTNorm(Tnorm):
    def conjunction(self, a, b):
        return torch.min(a, b)
