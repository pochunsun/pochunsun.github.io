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
We consider scalar field <img src="https://render.githubusercontent.com/render/math?math=\phi"> only dependent on holographic direction <img src="https://render.githubusercontent.com/render/math?math=z">

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle\mathcal{L}_{\text{matter}}=\sqrt{-\det (g)} \left(-\frac{1}{2} m^2 \phi ^2-V(\phi )-\frac{1}{2} \left(\frac{\partial \phi }{\partial z}\right)^2\right)\end{equation*}">

in asymptotic AdS black hole background

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle ds^2=\frac{1}{z^2}\left(-h(z)dt^2+\frac{dz^2}{h(z)}+\sum _{i=1}^n dx_i^2\right)\end{equation*}">

where emblackening function have following properties

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle h(0)=1\qquad\text{and}\qquad h(1)=0\qquad\left(\text{For simplicity, we set}\quadz_h=1\right) \end{equation*}">

Specially,

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle h(z)=1-z^3-Q^2 z^3+Q^2 z^4\end{equation*}">

in RN case. Note that, in extremal case, <img src="https://render.githubusercontent.com/render/math?math=Q=\sqrt{3}">.


## Reproduced Metric and EoM

The EoM for <img src="https://render.githubusercontent.com/render/math?math=\phi(z)"> is

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle z^2 h(z) \phi^{''}(z)+\left(z^2 h^{'}(z)-2 z h(z)\right) \phi^{'}(z)-m^2 \phi -\frac{\delta V(\phi )}{\delta \phi }=0\end{equation*}">

Now we're wanna let <img src="https://render.githubusercontent.com/render/math?math=g_{11}=1">. Consider the following coordinate transformation

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle d\eta=-\frac{dz}{z \sqrt{h(z)}}\quad ,\qquad \eta =\int _z^{z_h=1}\frac{dz}{z \sqrt{h(z)}}\end{equation*}">

Then the EoM become

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle \frac{\partial \Pi }{\partial \eta }+  H_R(\eta )\Pi-m^2 \phi-\frac{\delta V(\phi )}{\delta \phi }=0\quad ,\qquad H_R(\eta )\equiv\frac{6 h(z(\eta ))-y h^{'}(z(\eta ))}{2 \sqrt{h(z(\eta ))}}\end{equation*}">

where <img src="https://render.githubusercontent.com/render/math?math=\Pi:=\frac{\partial \phi }{\partial \eta }">. Specially,  for Schwarzschild case, 

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle z=\text{sech}^{\frac{2}{3}}\left(\frac{3 \eta }{2}\right)\quad ,\qquad H_R(\eta )=3 \coth (3 \eta )\end{equation*}">

```python
def h(y):
    return 1-y**3-(q**2)*y**3+(q**2)*y**4

def eta_coord(y):
    r=[]
    for i in range (0,len(y)):
        r=np.append(r,integrate.quad(lambda z: 1/(z*(h(z)**(1/2))), y[i],1)[0])
    return r

def y_coord(eta):
    r=[]
    accep_error=10**(-3)
    for i in range (0,len(eta)):
        erroreta=100
        y_lower=0
        y_upper=1
        while abs(erroreta) > accep_error:
            yy=(y_lower+y_upper)/2
            test_eta=eta_coord(np.array([yy]))
            if test_eta > eta[i] :
                 y_lower=yy
            else: 
                 y_upper=yy
            erroreta=eta[i]-test_eta
        r=np.append(r,yy)  
    return r

def H_r(eta):
    eta=eta/scale
    return (6*h(y_coord(eta))-y_coord(eta)*derivative(h,y_coord(eta), dx=1e-6))/(2*(h(y_coord(eta))**(1/2)))

def v(phi):
    return (lam*phi**4)/4 #delta_v(phi)-> derivative(v,phi)

def ff(eta,phi,pi):
    return (derivative(v,phi, dx=1e-6)-H_r(eta*scale)*pi*scale+phi*m2)*scale**2


eta_base=[]
for i in range(0,int(layer)):
    eta_base.append(ir_cutoff+i*abs(delta_eta))
tanh = nn.Tanh()
print(len(eta_base),eta_base[layer-1],layer)
```
```python
xx=np.arange(0.1, 1.05/scale, 0.05)
plt.plot(xx,3*np.cosh(3*xx/scale)/np.sinh(3*xx/scale), lw=2, label='$3 coth(3\eta)$')
plt.plot(xx,H_r(xx), lw=2, label='$H_R(\eta)$')
plt.title('Reproduced Metric $H(\eta)$')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('$\eta$')
plt.ylabel('$H_R(\eta)$')
#plt.tight_layout()
#plt.savefig("Hr_rnq09_n3.png")
plt.show()
```
<figure>
<a href="https://live.staticflickr.com/65535/51222259119_b1b528594e.jpg"><img src="https://live.staticflickr.com/65535/51222259119_b1b528594e.jpg" alt="" width="500"></a>
</figure>

## Activation Function
### _Runge-Kutta Fourth-Order_
From EoM, let's say

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle \tilde{f}(\eta ,\phi ,\Pi )\equiv \frac{\partial \Pi }{\partial \eta }=-H_R\Pi+m^2 \phi+\frac{\delta V(\phi )}{\delta \phi }\end{equation*}">

The activation function at each layer is

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle \phi (\Delta \eta +\eta )=\phi (\eta )+\Delta \eta  \left(\Pi (\eta )+\frac{1}{3} \left(k_1+k_2+k_3\right)\right)\end{equation*}">

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle \Pi (\Delta \eta +\eta )=\Pi (\eta )+\frac{1}{3} \left(k_1+2 k_2+2 k_3+k_4\right)\end{equation*}">

where $k_1, k_2, k_3, k_4$ are defined by

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle k_1\equiv\frac{\Delta \eta }{2}  \tilde{f}(\eta ,\phi ,\Pi )\end{equation*}">

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle k_2\equiv\frac{\Delta \eta}{2}   \tilde{f}\left(\eta+\frac{\Delta \eta }{2} ,\phi+k ,\Pi+k_1 \right),\qquad k\equiv\frac{\Delta \eta }{2}  \left(\Pi+\frac{k_1}{2} \right)\end{equation*}">

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle k_3\equiv\frac{\Delta \eta}{2}   \tilde{f}\left(\eta+\frac{\Delta \eta }{2} ,\phi+k ,\Pi+k_2 \right)\end{equation*}">

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle k_4\equiv\frac{\Delta \eta}{2}   \tilde{f}\left(\eta %2B\Delta \eta ,\phi %2B\ell  ,\Pi %2B 2 k_3 \right),\qquad\ell \equiv\Delta \eta  \left(k_3 %2B \Pi \right)\end{equation*}">




[1] K. Hashimoto, S. Sugishita, A. Tanaka and A. Tomiya, *Deep Learning and AdS/CFT,* [*Phys. Rev. D* **98**, 106014 (2018)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.046019)
