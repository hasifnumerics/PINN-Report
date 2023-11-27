This repository was created to showcase an independent research report on the concept of Physics Informed Neural Networks (PINNs). Here I explain briefly the concepts of Backward Propagation, Forward Propagation, Loss Function formulation, optimization etc. 

I then demonstrate a PiNNs approach to approximating the solutions to the 1D heat equation using PyTorch and compare the behavior of solutions based on the number of training steps.

# Physics-Informed Neural Networks for 1D Heat Equation

We aim to solve the 1D heat equation using Physics-Informed Neural Networks. The mathematical model we are considering is:

**The 1D heat equation is given by:**

`u_t = u_{xx}`

Where:
- `u_t` denotes the partial derivative of `u` with respect to `t`.
- `u_{xx}` denotes the second partial derivative of `u` with respect to `x`.

**Initial and Boundary Conditions**

The initial condition is:

`u(x, 0) = sin(πx)`

And the boundary conditions are:

- At `x = 0`: `u(0, t) = 0`
- At `x = 1`: `u(1, t) = 0`

![Alt text] (https://github.com/hasifnumerics/PINN-Report/blob/6455dc650ae0f9ab7a1b3537dc49583f4bafca21/1d%20heat%2015000%20deal%20(1).png)
