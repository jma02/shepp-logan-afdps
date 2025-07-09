# This code is authored by Jonathan Ma, johnma@udel.edu
# This is to say most of the other code isn't, or was taken and modified.
import abc
import torch
import numpy as np
from sdes.sde import SDE

"""
 The EDM paper uses the general SDE
 dx = \frac{\dot{s}(t)}{s(t)}x\,dt + s(t)\sqrt{2\dot{\sigma}(t) \sigma(t)}\,dw_t.

 The AFDPS authors and the EDM authors both
  use $\sigma(t) = t$, and $s(t) = 1$ identically.

 This corresponds to the DDIM SDE.
 """
class EDMSDE(SDE):
  def __init__(self, N=1000):
    """Construct the SDE described in the EDM paper.

    Args:
      N: number of discretization steps
    """
    super().__init__(N)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    drift = torch.zeros_like(x)
    diffusion = torch.sqrt(2*t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    # TODO: Find the closed form formula for this
    # return mean, std
    pass

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    # TODO: Find the closed form formula for this
    # return logps
    pass

  """
  # Change if needed, defaults to Euler-Maruyama discretization
  def discretize(self, x, t):
    
    return f, G
  """