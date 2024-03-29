---
layout: post 
title: Learning Holographic Metric by Experimental Data (Ⅱ)
description: "AdS/CFT is a useful mathematical tool to solve field theory problems by computing gravitational calculations. Here we provide a way to build up the continuous bulk metric by experimental data from field theory."
tags: [Deep Learning,Pytorch,Holography,AdS/CFT,Entanglement Entropy,Quantum Information]
image:
  background: triangular.png
  
modified: 2021-06-03
---

## Assumptions of Reproduced Metric <img src="https://render.githubusercontent.com/render/math?math=%24%5CLarge%20H_R(%5Ceta%20)%24">

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0AH_R(%5Ceta%20)%5Cin%20%5Cmathcal%7BC%7D_%7B%5Cinfty%20%7D%5Cqquad%5Cforall%20%5Ceta%20%5Cgeq%200%5Cqquad%5Cqquad%5Cqquad(A1)%0A%5Cend%7Bequation*%7D">

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0AH_R(%5Ceta%20)%5Cto%20%5Cfrac%7B1%7D%7B%5Ceta%20%7D%5Cquad%5Ctext%7Bas%7D%5Cquad%5Ceta%5Cto%200%5Cqquad%5Cqquad%5Cqquad(A2)%0A%5Cend%7Bequation*%7D">

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0AH_R(%5Ceta%20)%5Cto%20(n%2B1)%5Cquad%5Ctext%7Bas%7D%5Cquad%5Ceta%5Cto%20%5Cinfty%5Cquad(%5Ctext%7Bin%7D%5C%2C(n%2B2)%5Ctext%7B-dimensional%20spacetime%7D)%5Cqquad%5Cqquad%5Cqquad(A3)%0A%5Cend%7Bequation*%7D">

Thus, we are able to say

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0AH_R(%5Ceta%20)%3DH%5E%7B%5Ctext%7Bir%7D%7D_R(%5Ceta%20)%5Cequiv%5Cfrac%7Bc_%7B-1%7D%7D%7B%5Ceta%20%7D%2Bc_0%2Bc_1%20%5Ceta%20%2Bc_2%20%5Ceta%20%5E2%2B%5Ctext%7B...%7D%3D%5Coverset%7B%5Crightharpoonup%20%7D%7Bc%7D.%5Coverset%7B%5Crightharpoonup%20%7D%7B%5Ceta%20%7D_%7B%5Ctext%7Bir%7D%7D%5Cquad%5Ctext%7Bas%7D%5Cquad%5Ceta%20%3C%5Ceta%20%5E*%0A%5Cend%7Bequation*%7D">

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0AH_R(%5Ceta%20)%3DH%5E%7B%5Ctext%7Buv%7D%7D_R(%5Ceta%20)%5Cequiv%20d_0%2Bd_1%20%5Ceta%20%2Bd_2%20%5Ceta%20%5E2%2B%5Ctext%7B...%7D%3D%5Coverset%7B%5Crightharpoonup%20%7D%7Bd%7D.%5Coverset%7B%5Crightharpoonup%20%7D%7B%5Ceta%20%7D_%7B%5Ctext%7Buv%7D%7D%5Cquad%5Ctext%7Bas%7D%5Cquad%5Ceta%20%3E%5Ceta%20%5E*%0A%5Cend%7Bequation*%7D">

where <img src="https://render.githubusercontent.com/render/math?math=%24%5Ceta%5E*%24"> is a matching point. (<img src="https://render.githubusercontent.com/render/math?math=%24%5Ceta%5E*%3D0.5%24"> in our code) Due to (A2),

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0Ac_%7B-1%7D%3D1%20%5Cqquad%5Ctext%7Band%7D%5Cqquad%20c_0%3D0%0A%5Cend%7Bequation*%7D">

From (A3), we can fix the two of coefficients <img src="https://render.githubusercontent.com/render/math?math=%24%5Coverset%7B%5Crightharpoonup%20%7D%7Bd%7D%3D(d_0%2C%20d_1%2C%20d_2%2C%20...)%24">. Moreover, because of (A1), we have the matching conditions

<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0AH%5E%7B%5Ctext%7Bir%7D%7D_R%5Cleft(%5Ceta%20%5E*%5Cright)%3DH%5E%7B%5Ctext%7Buv%7D%7D_R%5Cleft(%5Ceta%20%5E*%5Cright)%2C%5Cquad%5Cfrac%7B%5Cpartial%20H%5E%7B%5Ctext%7Bir%7D%7D_R%5Cleft(%5Ceta%20%5E*%5Cright)%7D%7B%5Cpartial%20%5Ceta%20%7D%3D%5Cfrac%7B%5Cpartial%20H%5E%7B%5Ctext%7Buv%7D%7D_R%5Cleft(%5Ceta%20%5E*%5Cright)%7D%7B%5Cpartial%20%5Ceta%20%7D%2C%5Ctext%7B...%7D%5Ctext%7B...%7D%0A%5Cend%7Bequation*%7D">

---

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
---

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
```python
did_p=100*uv_cutoff
interval=(ini_eta-ir_cutoff)/did_p
eta=ir_cutoff
fitting_results_exp_h=[]
fitting_results_exp_eta=[]
for i in range(0,int(did_p+1)):
    fitting_results_exp_eta.append(eta)
    fitting_results_exp_h.append(fitting_results_exp(h_fitting_vec_tensor_ir,eta))
    eta+=interval

interval=(ini_eta-ir_cutoff)/did_p
eta=ir_cutoff
before_h=[]
before_eta=[]
for i in range(0,int(did_p+1)):
    before_eta.append(eta)
    before_h.append(fitting_results_exp(torch.tensor(h_fitting_vec_ir),eta))
    eta+=interval
```

```python
plt.plot(loss_epoch, lw=2, label='Loss Function')
plt.title('Time Evolution of Loss Function')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('epoch')
plt.ylabel('Loss Function')
plt.tight_layout()
plt.savefig("loss_rnq09_3n_1.png")
plt.show()
```

<figure>
<a href="https://live.staticflickr.com/65535/51221809556_0bc1bfceef_w.jpg"><img src="https://live.staticflickr.com/65535/51221809556_0bc1bfceef_w.jpg" alt="" width="500"></a>
</figure>

```python
plt.plot(np.array(eta_base),H_r(np.array(eta_base)), lw=5, label='True Metric')
plt.plot(fitting_results_exp_eta,fitting_results_exp_h, lw=5, label='After Learning')
plt.plot(before_eta,before_h, lw=5, label='Before Learning')
plt.title('Learning Metric')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.xlabel('$\eta$')
plt.ylabel('$h(\eta)$')
plt.tight_layout()
plt.savefig("learning_rnq09_3n_1.png")
plt.show()
```
<figure>
<a href="https://live.staticflickr.com/65535/51222028018_2fec228443_w.jpg"><img src="https://live.staticflickr.com/65535/51222028018_2fec228443_w.jpg" alt="" width="500"></a>
</figure>

<a href="{{ site.url }}/Learning-Holographic-Metric-by-Experimental-Data-(Ⅰ)/" class="btn btn-info">Learning Holographic Metric by Experimental Data (Ⅰ)</a>

[1] K. Hashimoto, S. Sugishita, A. Tanaka and A. Tomiya, *Deep Learning and AdS/CFT,* [*Phys. Rev. D* **98**, 106014 (2018)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.98.046019)
