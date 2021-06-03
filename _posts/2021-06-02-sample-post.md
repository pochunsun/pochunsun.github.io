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

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle ds^2=\frac{1}{z^2}\left(-h(z)dt^2 %2B\frac{dz^2}{h(z)} %2B\sum _{i=1}^n dx_i^2\right)\end{equation*}">

where emblackening function have following properties

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle h(0)=1\qquad\text{and}\qquad h(1)=0\qquad\left(\text{For simplicity, we set}\quadz_h=1\right) \end{equation*}">

Specially,

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle h(z)=1-z^3-Q^2 z^3 %2B Q^2 z^4\end{equation*}">

in RN case. Note that, in extremal case, <img src="https://render.githubusercontent.com/render/math?math=Q=\sqrt{3}">.


## Reproduced Metric and EoM

The EoM for <img src="https://render.githubusercontent.com/render/math?math=\phi(z)"> is

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle z^2 h(z) \phi^{''}(z) %2B \left(z^2 h^{'}(z)-2 z h(z)\right) \phi^{'}(z)-m^2 \phi -\frac{\delta V(\phi )}{\delta \phi }=0\end{equation*}">

Now we're wanna let <img src="https://render.githubusercontent.com/render/math?math=g_{11}=1">. Consider the following coordinate transformation

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle d\eta=-\frac{dz}{z \sqrt{h(z)}}\quad ,\qquad \eta =\int _z^{z_h=1}\frac{dz}{z \sqrt{h(z)}}\end{equation*}">

Then the EoM become

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle \frac{\partial \Pi }{\partial \eta } %2B  H_R(\eta )\Pi-m^2 \phi-\frac{\delta V(\phi )}{\delta \phi }=0\quad ,\qquad H_R(\eta )\equiv\frac{6 h(z(\eta ))-y h^{'}(z(\eta ))}{2 \sqrt{h(z(\eta ))}}\end{equation*}">

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

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle \tilde{f}(\eta ,\phi ,\Pi )\equiv \frac{\partial \Pi }{\partial \eta }=-H_R\Pi %2B m^2 \phi %2B\frac{\delta V(\phi )}{\delta \phi }\end{equation*}">

The activation function at each layer is

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle \phi (\Delta \eta %2B\eta )=\phi (\eta ) %2B\Delta \eta  \left(\Pi (\eta ) %2B\frac{1}{3} \left(k_1 %2B k_2 %2B k_3\right)\right)\end{equation*}">

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle \Pi (\Delta \eta %2B\eta )=\Pi (\eta ) %2B\frac{1}{3} \left(k_1 %2B 2 k_2 %2B 2 k_3 %2B k_4\right)\end{equation*}">

where <img src="https://render.githubusercontent.com/render/math?math=%24k_1%2C%20k_2%2C%20k_3%2C%20k_4%24"> are defined by

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle k_1\equiv\frac{\Delta \eta }{2}  \tilde{f}(\eta ,\phi ,\Pi )\end{equation*}">

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle k_2\equiv\frac{\Delta \eta}{2}   \tilde{f}\left(\eta %2B \frac{\Delta \eta }{2} ,\phi %2B k ,\Pi %2B k_1 \right),\qquad k\equiv\frac{\Delta \eta }{2}  \left(\Pi %2B \frac{k_1}{2} \right)\end{equation*}">

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle k_3\equiv\frac{\Delta \eta}{2}   \tilde{f}\left(\eta %2B\frac{\Delta \eta }{2} ,\phi %2B k ,\Pi %2B k_2 \right)\end{equation*}">

<img src="https://render.githubusercontent.com/render/math?math=\begin{equation*}\displaystyle k_4\equiv\frac{\Delta \eta}{2}   \tilde{f}\left(\eta %2B\Delta \eta ,\phi %2B\ell  ,\Pi %2B 2 k_3 \right),\qquad\ell \equiv\Delta \eta  \left(k_3 %2B \Pi \right)\end{equation*}">

```python
phi_list_pos_exp=[]
pi_list_pos_exp=[]
phi_list_neg_exp=[]
pi_list_neg_exp=[]
phi_exp_data=[]
pi_exp_data=[]
train_data_y=[]
np.random.seed(1)
while len(phi_list_pos_exp)<num_training_data or len(phi_list_neg_exp)<num_training_data :
        eta=uv_cutoff
        phi_ini=np.random.uniform(low=0, high=1.7, size=(num_training_data*50)) #random.uniform(0,1)
        pi_ini=np.random.uniform(low=-0.2, high=0.7, size=(num_training_data*50))*scale#random.uniform(-1.7,0.01)
        phi=phi_ini
        pi=pi_ini
        for i in range(0,int(layer-1)):
            k1=delta_eta*ff(np.array([eta]),phi,pi)/2
            k=delta_eta*(k1/2+pi)/2
            k2=delta_eta*ff(np.array([eta+delta_eta/2]),phi+k,pi+k1)/2
            k3=delta_eta*ff(np.array([eta+delta_eta/2]),phi+k,pi+k2)/2
            ell=delta_eta*(pi+k3)
            k4=delta_eta*ff(np.array([eta+delta_eta]),phi+k,pi+2*k3)/2
            
            phi_new=phi+delta_eta*(pi+(k1+k2+k3)/3)
            pi_new=pi+(k1+2*k2+2*k3+k4)/3
            eta+=delta_eta
            phi=phi_new
            pi=pi_new
        for j in range(0,len(phi)):
            final_layer=t(pi[j])
            if final_layer>0.5: #2*pi/eta-m2*phi-delta_v(phi)
                if len(phi_list_neg_exp)<num_training_data:
                    phi_list_neg_exp.append(phi_ini[j])
                    pi_list_neg_exp.append(pi_ini[j])
                    phi_exp_data.append(phi_ini[j])
                    pi_exp_data.append(pi_ini[j])
                    train_data_y.append(final_layer) #false t(pi)=1
            else: 
                if len(phi_list_pos_exp)<num_training_data:
                    phi_list_pos_exp.append(phi_ini[j])
                    pi_list_pos_exp.append(pi_ini[j])
                    phi_exp_data.append(phi_ini[j])
                    pi_exp_data.append(pi_ini[j])
                    train_data_y.append(final_layer) #true t(pi)=0
                    #print(len(train_data_y))
```
## Generating Data by Real Metric
Set <img src="https://render.githubusercontent.com/render/math?math=%24%5Cepsilon%20%3D0.1%24">

True: <img src="https://render.githubusercontent.com/render/math?math=%24%5CPi%20%5Cleq%20%5Cepsilon%24">     <img src="https://render.githubusercontent.com/render/math?math=%24%5Cqquad%24">       False: <img src="https://render.githubusercontent.com/render/math?math=%24%5CPi%20%3E%5Cepsilon%24">

<figure>
<a href="https://live.staticflickr.com/65535/51222672590_818823e7f9.jpg"><img src="https://live.staticflickr.com/65535/51222672590_818823e7f9.jpg" alt="" width="500"></a>
</figure>

## Flowchart

<figure>
<a href="https://live.staticflickr.com/65535/51221599581_8f6f9ec585_w.jpg"><img src="https://live.staticflickr.com/65535/51221599581_8f6f9ec585_w.jpg" alt="" width="500"></a>
</figure>

## Assumptions of Reproduced Metric <img src="https://render.githubusercontent.com/render/math?math=%24%5CLarge%20H_R(%5Ceta%20)%24">

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0AH_R(%5Ceta%20)%5Cin%20%5Cmathcal%7BC%7D_%7B%5Cinfty%20%7D%5Cqquad%5Cforall%20%5Ceta%20%5Cgeq%200%5Cqquad%5Cqquad%5Cqquad(A1)%0A%5Cend%7Bequation*%7D">

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0AH_R(%5Ceta%20)%5Cto%20%5Cfrac%7B1%7D%7B%5Ceta%20%7D%5Cquad%5Ctext%7Bas%7D%5Cquad%5Ceta%5Cto%200%5Cqquad%5Cqquad%5Cqquad(A2)%0A%5Cend%7Bequation*%7D">

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0AH_R(%5Ceta%20)%5Cto%20(n%2B1)%5Cquad%5Ctext%7Bas%7D%5Cquad%5Ceta%5Cto%20%5Cinfty%5Cquad(%5Ctext%7Bin%7D%5C%2C(n%2B2)%5Ctext%7B-dimensional%20spacetime%7D)%5Cqquad%5Cqquad%5Cqquad(A3)%0A%5Cend%7Bequation*%7D">

Thus, we are able to say

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0AH_R(%5Ceta%20)%3DH%5E%7B%5Ctext%7Bir%7D%7D_R(%5Ceta%20)%5Cequiv%5Cfrac%7Bc_%7B-1%7D%7D%7B%5Ceta%20%7D%2Bc_0%2Bc_1%20%5Ceta%20%2Bc_2%20%5Ceta%20%5E2%2B%5Ctext%7B...%7D%3D%5Coverset%7B%5Crightharpoonup%20%7D%7Bc%7D.%5Coverset%7B%5Crightharpoonup%20%7D%7B%5Ceta%20%7D_%7B%5Ctext%7Bir%7D%7D%5Cquad%5Ctext%7Bas%7D%5Cquad%5Ceta%20%3C%5Ceta%20%5E*%0A%5Cend%7Bequation*%7D">

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0AH_R(%5Ceta%20)%3DH%5E%7B%5Ctext%7Buv%7D%7D_R(%5Ceta%20)%5Cequiv%20d_0%2Bd_1%20%5Ceta%20%2Bd_2%20%5Ceta%20%5E2%2B%5Ctext%7B...%7D%3D%5Coverset%7B%5Crightharpoonup%20%7D%7Bd%7D.%5Coverset%7B%5Crightharpoonup%20%7D%7B%5Ceta%20%7D_%7B%5Ctext%7Buv%7D%7D%5Cquad%5Ctext%7Bas%7D%5Cquad%5Ceta%20%3E%5Ceta%20%5E*%0A%5Cend%7Bequation*%7D">

where <img src="https://render.githubusercontent.com/render/math?math=%24%5Ceta%5E*%24"> is matching point. (<img src="https://render.githubusercontent.com/render/math?math=%24%5Ceta%5E*%3D0.5%24"> in our code) Due to (A2),

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0Ac_%7B-1%7D%3D1%20%5Cqquad%5Ctext%7Band%7D%5Cqquad%20c_0%3D0%0A%5Cend%7Bequation*%7D">

From (A3), we can fix two of coefficient <img src="https://render.githubusercontent.com/render/math?math=%24%5Coverset%7B%5Crightharpoonup%20%7D%7Bd%7D%3D(d_0%2C%20d_1%2C%20d_2%2C%20...)%24">. Moreover, because of (A1), we have matching conditions

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0AH%5E%7B%5Ctext%7Bir%7D%7D_R%5Cleft(%5Ceta%20%5E*%5Cright)%3DH%5E%7B%5Ctext%7Buv%7D%7D_R%5Cleft(%5Ceta%20%5E*%5Cright)%2C%5Cquad%5Cfrac%7B%5Cpartial%20H%5E%7B%5Ctext%7Bir%7D%7D_R%5Cleft(%5Ceta%20%5E*%5Cright)%7D%7B%5Cpartial%20%5Ceta%20%7D%3D%5Cfrac%7B%5Cpartial%20H%5E%7B%5Ctext%7Buv%7D%7D_R%5Cleft(%5Ceta%20%5E*%5Cright)%7D%7B%5Cpartial%20%5Ceta%20%7D%2C%5Ctext%7B...%7D%5Ctext%7B...%7D%0A%5Cend%7Bequation*%7D">

```python
ReLU= nn.ReLU()
from torch.autograd import grad
def factorial(n):
    if n == 0:
        return 1
    else:
        return n*factorial(n-1)

def D(f,x,n):
    x=torch.tensor([float(x)], requires_grad=True)
    func=f(x)
    for i in range(n):
        grads=grad(func, x, create_graph=True)[0]
        func=grads
    return grads
def DP(x,power,n):
    if n>power and power>=0:
        return torch.tensor([0.])
    x=torch.tensor([float(x)], requires_grad=True)
    if n==0:
        return P(power,x)
    else:
        func=P(power,x)
        for i in range(n):
            grads=grad(func, x, create_graph=True)[0]
            func=grads
        return grads
    
def D_fitting_results_exp(fitting_results_exp,h_fitting_vec_tensor_ir,x,n):
    x=torch.tensor([float(x)], requires_grad=True)
    func=fitting_results_exp(h_fitting_vec_tensor_ir,x)
    for i in range(n):
        grads=grad(func, x, create_graph=True)[0]
        func=grads
    return grads
```
```python
def fitting_results_exp(h_fitting_vec_tensor_ir,x):
    x=x/scale
    def h_bhk_uv(eta):
        def h_bhk_uv_sub(eta):
            summ=P(0,eta)*h_fitting_vec_tensor_ir[0]
            for i in range(1,int(n_ir)):
                summ+=P(i,eta)*h_fitting_vec_tensor_ir[i]
            return summ
        return h_bhk_uv_sub(eta)
        
    def h_bhk_ir(eta):
        #A_matrix
        A_mat=torch.zeros(n_ir, n_ir, dtype=torch.float64)
        for i in range(0,n_ir):
            for j in range(0,n_ir):
                A_mat[i][j]=A_mat[i][j]+DP(ma_pt,j+1,i)
        A_inverse=torch.inverse(A_mat)
            
        #b_vec
        b_vec=torch.zeros(n_ir, 1, dtype=torch.float64)
        b_vec[0]=b_vec[0]+h_bhk_uv(ma_pt)-1/ma_pt
        for i in range(1,n_ir):
            b_vec[i]=b_vec[i]+D(h_bhk_uv,ma_pt,i)-DP(ma_pt,-1,i)

        #A_matrix**(-1) dot b_vec
        x_vec=torch.mm(A_inverse,b_vec)

        r=1/eta
        for i in range(0,n_ir):
            r+=x_vec[i]*P(i+1,eta)
        return r
    
    if eta<ma_pt:
        return h_bhk_ir(x)
    else:
        return h_bhk_uv(x)
```
```python
n_ir=2
#n_uv=8
ma_pt=ini_eta/2#(ini_eta-ir_cutoff)/2
#np.random.seed(7)
avr=random.uniform(-1,1)
h_fitting_vec_ir=[]
for i in range(0,int(n_ir)):
    h_fitting_vec_ir.append([np.random.randn()+avr])#np.random.randn()
h_fitting_vec_tensor_ir=torch.tensor(h_fitting_vec_ir, requires_grad=True)
print(h_fitting_vec_tensor_ir,len(h_fitting_vec_tensor_ir))
def P(n,x):
    return  x**n

class PyTorchLinearRegression_fitting(nn.Module):
    def __init__(self):
        super().__init__()
        self.training_data_ir = nn.Parameter(h_fitting_vec_tensor_ir)
        
    def forward(self,eta,phi,pi):
        
        def ff_tensor(eta,phi,pi):
            return (derivative(v,phi, dx=1e-6)-fitting_results_exp(self.training_data_ir,eta*scale)*pi*scale+phi*m2)*scale**2
        
        def t_tensor(F):
            return 1-torch.exp(-(torch.abs(F)**2.5)/0.03) #ReLU(F-ir_cutoff)+ReLU(-F-ir_cutoff) #(tanh(wi*(F-0.1))-tanh(wi*(F+0.1))+2)/2 #(F**2)*wi
        
        def ir_reg(order,coe,training_data_ir):
            r=0
            for j in range(0,int(order+1)):
                the=factorial(j)*((-1)**j)*(1/(ir_cutoff)**(1+j))
                if j==0:
                    exp=training_data_ir[0]
                else:
                    exp=D_fitting_results_exp(fitting_results_exp,training_data_ir,ir_cutoff,j)
                r+=coe*ReLU(abs(the-exp)-1/(ir_cutoff)**(j))**2 #ReLU(abs(the-exp)-1/(ir_cutoff)**(j))**2
            return r

        for i in range(0,layer-1):
            k1=delta_eta*ff_tensor(eta,phi,pi)/2
            k=delta_eta*(k1/2+pi)/2
            k2=delta_eta*ff_tensor(eta+delta_eta/2,phi+k,pi+k1)/2
            k3=delta_eta*ff_tensor(eta+delta_eta/2,phi+k,pi+k2)/2
            ell=delta_eta*(pi+k3)
            k4=delta_eta*ff_tensor(eta+delta_eta,phi+k,pi+2*k3)/2

            phi_new=phi+delta_eta*(pi+(k1+k2+k3)/3)
            pi_new=pi+(k1+2*k2+2*k3+k4)/3
            eta+=delta_eta
            phi=phi_new
            pi=pi_new
        #2*pi/eta-m2*phi-delta_v(phi)
        order=0
        coe=0.1
        return (t_tensor(pi),torch.sum(self.training_data_ir**2),ir_reg(order,coe,self.training_data_ir))
    
model_fitting = PyTorchLinearRegression_fitting()#.to(device)
print(model_fitting.state_dict())
```
## Optimizer: Adam
<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0A%5Chat%7Bm%7D_%7Bi%2B1%7D%5Cleftarrow%20%5Cfrac%7B%5Cleft(1-%5Cbeta%20_1%5Cright)%20%5Cnabla%20_%7B%5Coverset%7B%5Crightharpoonup%20%7D%7Bd%7D%7DL_i%2B%5Cbeta%20_1%20%5Cleft(1-%5Cbeta%20_1%5E%7Bi%2B1%7D%5Cright)%20%5Chat%7Bm%7D_i%7D%7B1-%5Cbeta%20_1%5E%7Bi%2B1%7D%7D%0A%5Cend%7Bequation*%7D">

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0A%5Chat%7Bv%7D_%7Bi%2B1%7D%5Cleftarrow%20%5Cfrac%7B%5Cleft(1-%5Cbeta%20_2%5Cright)%20%5Cleft(%5Cnabla%20_%7B%5Coverset%7B%5Crightharpoonup%20%7D%7Bd%7D%7DL_i%5Cright)%7B%7D%5E2%2B%5Cbeta%20_2%20%5Cleft(1-%5Cbeta%20_2%5E%7Bi%2B1%7D%5Cright)%20%5Chat%7Bv%7D_i%7D%7B1-%5Cbeta%20_2%5E%7Bi%2B1%7D%7D%0A%5Cend%7Bequation*%7D">

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0A%5Coverset%7B%5Crightharpoonup%20%7D%7Bd%7D_%7Bi%2B1%7D%5Cleftarrow%20%5Coverset%7B%5Crightharpoonup%20%7D%7Bd%7D_i-%5Cfrac%7B%5Chat%7Bm%7D_i%20%5Ceta%20_%7B%5Cell%20%7D%7D%7B%5Cdelta%20%2B%5Csqrt%7B%5Chat%7Bv%7D_i%7D%7D%0A%5Cend%7Bequation*%7D">

```python
lr_xp_fitting=0.03 #0.03
lair=0.1
irc=0
cm=10
uvd=0.1
optimizer_exp_fitting=optim.Adam(model_fitting.parameters(), lr=lr_xp_fitting) #RMSprop #Adam
MSELoss = nn.MSELoss()
def build_train_step_exp_fitting(model, loss_fn, optimizer):
    
    def train_step_exp_fitting(eta,phi,pi, y):   
        
        #Training Model
        model_fitting.train()
        
        # Prediction
        yhat = model_fitting(eta,phi,pi)
        
        #Computing Loss Function
        
        loss = loss_fn(y, yhat[0])+lair*yhat[1]+yhat[2]
        
        #Computing Gradient
        loss.backward()
        
        #Optimization
        optimizer.step() 
        optimizer.zero_grad()
        return loss.item() 
    return train_step_exp_fitting
train_step_exp_fitting = build_train_step_exp_fitting(model_fitting, MSELoss, optimizer_exp_fitting)
print(model_fitting.state_dict())
```
```python
from torch.utils.data import Dataset
class SLRDataset(Dataset):
    def __init__(self, phi_exp_data_tensor, pi_exp_data_tensor, train_data_y_tensor):
        self.phi = phi_exp_data_tensor
        self.pi = pi_exp_data_tensor
        self.train = train_data_y_tensor
        
    def __getitem__(self, index):
        return (self.phi[index],self.pi[index], self.train[index])
    def __len__(self):
        return len(self.phi)
#
phi_exp_data_tensor = torch.from_numpy(np.array(phi_exp_data))
pi_exp_data_tensor = torch.from_numpy(np.array(pi_exp_data))
train_data_y_tensor = torch.from_numpy(np.array(train_data_y))
training_data = SLRDataset(phi_exp_data_tensor,pi_exp_data_tensor, train_data_y_tensor)

from torch.utils.data import DataLoader
train_loader = DataLoader(dataset=training_data, batch_size=50, shuffle=True)
```
```python
epochs=10
loss_epoch=[]
for epoch in range(epochs):
    losses_xp_fitting = []
    for phi_batch,pi_batch, train_batch in train_loader:
        loss = train_step_exp_fitting(ini_eta, phi_batch,pi_batch,train_batch)
        losses_xp_fitting.append(loss)
    loss_epoch.append(np.sum(losses_xp_fitting)/len(losses_xp_fitting))
    print('loss_epoch=',np.sum(losses_xp_fitting)/len(losses_xp_fitting))
print(model_fitting.state_dict())
```
```
loss_epoch= 12.358866596221924
loss_epoch= 9.669654583930969
loss_epoch= 7.773796164989472
loss_epoch= 6.478020656108856
loss_epoch= 5.501032030582428
loss_epoch= 4.766412723064422
loss_epoch= 4.42119995355606
loss_epoch= 4.286863183975219
loss_epoch= 4.241568195819855
loss_epoch= 4.228258776664734
OrderedDict([('training_data_ir', tensor([[ 3.8222],
        [-0.8429]]))])
```


[1] K. Hashimoto, S. Sugishita, A. Tanaka and A. Tomiya, *Deep Learning and AdS/CFT,* [*Phys. Rev. D* **98**, 106014 (2018)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.046019)
