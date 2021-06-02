---
layout: post
title: Learning Holographic Dual Metric by Receiving Experimental Data
description: "Sample post."
tags: [Deep Learning,Pytorch,Holography,AdS/CFT,Entanglement Entropy,Quantum Information]
image:
  background: triangular.png
  
modified: 2021-02-02
---

## ***Implemented by Pytorch***

## Package
```python
import numpy as np
import random as random
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import animation
from celluloid import Camera
import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace
from torch import optim
from torchdiffeq import odeint_adjoint as odeint
from decimal import *
import numpy as np
from mpmath import *
from sympy import *
import scipy.integrate as integrate
import scipy.special as sc
from scipy.misc import derivative
from pynverse import inversefunc
import pandas as pd
from pandas import Series,DataFrame
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```
## Final Layer

```python
def t(F):
    signr= np.heaviside(F-ir_cutoff,0)
    signl= np.heaviside(-F-ir_cutoff,0)
    return  signr+signl
Fp= np.arange(-0.6, 0.6, 0.001) #step
plt.plot(Fp,t(Fp), lw=5, label='$t(F)$')
plt.title('Function of Final Layer')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('F')
plt.ylabel('$t(F)$')
#plt.tight_layout()
#plt.savefig("Tanh.png")
plt.show()
```
<figure>
<img src="https://github.com/pochunsun/pochunsun.github.io/blob/main/images/final%20layer.png" width="500" >
</figure>




[1] K. Hashimoto, S. Sugishita, A. Tanaka and A. Tomiya, *Deep Learning and AdS/CFT,* [*Phys. Rev. D* **98**, 106014 (2018)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.046019)
