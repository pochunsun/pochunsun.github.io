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
<a href="https://live.staticflickr.com/65535/51221043969_643c40812e.jpg"><img src="https://live.staticflickr.com/65535/51221043969_643c40812e.jpg" alt="" width="500"></a>
</figure>

## Setup
We consider scalar field $\phi$ only dependent on holographic direction $z$
$$ \tag{1}
\mathcal{L}_{\text{matter}}=\sqrt{-\det (g)} \left(-\frac{1}{2} m^2 \phi ^2-V(\phi )-\frac{1}{2} \left(\frac{\partial \phi }{\partial z}\right)^2\right)
$$

in asymptotic AdS black hole background
$$ \tag{2}
ds^2=\frac{1}{z^2}\left(-h(z)dt^2+\frac{dz^2}{h(z)}+\sum _{i=1}^n dx_i^2\right)
$$
where emblackening function have following properties
$$ \tag{3}
h(0)=1 \&   h(1)=0 \left(\text{For simplicity, we set } z_h=1\right)
$$

Specially, 
$$\tag{4}
h(z)=1-z^3-Q^2 z^3+Q^2 z^4
$$

in RN case. Note that, in extremal case, $Q=\sqrt{3}$. 

```math
SE = \frac{\sigma}{\sqrt{n}}
```

<img src="https://render.githubusercontent.com/render/math?math=  e^{i \pi} =-1">


<dl><dd><span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML"  alttext="{\displaystyle {\text{s}}=\int _{a}^{b}{\sqrt {\mathrm {d} x^{2}+\mathrm {d} y^{2}}}=\int _{a}^{b}{\sqrt {1+y'^{2}}}\,\mathrm {d} x,}">
  <semantics>

[1] K. Hashimoto, S. Sugishita, A. Tanaka and A. Tomiya, *Deep Learning and AdS/CFT,* [*Phys. Rev. D* **98**, 106014 (2018)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.046019)
