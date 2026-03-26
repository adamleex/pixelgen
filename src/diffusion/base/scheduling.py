import torch
from torch import Tensor

class BaseScheduler:
    def alpha(self, t) -> Tensor:
        ...
    def sigma(self, t) -> Tensor:
        ...

    # Raw value APIs preserve the input shape; useful for token-wise or spatial timestep tensors.
    def alpha_value(self, t) -> Tensor:
        ...

    def sigma_value(self, t) -> Tensor:
        ...

    def dalpha(self, t) -> Tensor:
        ...
    def dsigma(self, t) -> Tensor:
        ...

    def dalpha_value(self, t) -> Tensor:
        ...

    def dsigma_value(self, t) -> Tensor:
        ...

    def dalpha_over_alpha(self, t) -> Tensor:
        return self.dalpha(t) / self.alpha(t)

    def dsigma_mul_sigma(self, t) -> Tensor:
        return self.dsigma(t)*self.sigma(t)

    def drift_coefficient(self, t):
        alpha, sigma = self.alpha(t), self.sigma(t)
        dalpha, dsigma = self.dalpha(t), self.dsigma(t)
        return dalpha/(alpha + 1e-6)

    def diffuse_coefficient(self, t):
        alpha, sigma = self.alpha(t), self.sigma(t)
        dalpha, dsigma = self.dalpha(t), self.dsigma(t)
        return dsigma*sigma - dalpha/(alpha + 1e-6)*sigma**2

    def w(self, t):
        return self.sigma(t)
