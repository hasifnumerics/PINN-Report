# PINN-Report
\documentclass[12pt]{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\begin{document}

\title{Physics Informed Neural Networks: A New Approach to Scientific Computing}
\author{Hasif Ahmed}
\date{\today}
\maketitle

\begin{abstract}
Physics Informed Neural Networks (PINNs) are a novel approach that combines the power of deep learning and physical models to solve problems in science and engineering that are challenging for traditional solvers. This report provides a comprehensive overview of the mathematical foundations of neural networks and demonstrates how PINNs can be applied to a variety of problems in science. We provide a detailed analysis of the strengths and limitations of PINNs and compare them with traditional solvers.
\end{abstract}

\section{Introduction}
The advancement in artificial intelligence and machine learning has opened up new avenues for solving complex problems in science and engineering. Traditional solvers, based on numerical methods and physical models, have limitations in terms of computational efficiency and accuracy, especially for high-dimensional and nonlinear problems. Physics Informed Neural Networks (PINNs) are a novel approach that leverages the power of deep learning to improve the solution of such problems. Recent breakthroughs in machine learning, spurred by vast data and computational power, have impacted various fields like image recognition, natural language processing, cognitive science, and genomics. However, many analysis tasks, especially in physical, biological, or engineering systems, grapple with limited data, making data acquisition costly. In such scenarios, conventional machine learning models, like deep or convolutional neural networks, often lack robustness and don't guarantee convergence. Training such models to understand high-dimensional input-output pairs with limited data seems impractical. However, when analyzing physical or biological systems, there's significant prior knowledge not utilized in today's machine learning. Whether it's consistent physical laws, empirical rules, or domain expertise, this knowledge can regularize and guide models, e.g., by eliminating non-realistic solutions in fluid dynamics that defy mass conservation. Integrating this structured knowledge into algorithms enhances the data's informational value, allowing models to hone in on correct solutions even with minimal training data. This report aims to provide a comprehensive overview of the mathematical foundations of neural networks, explain the concept of PINNs, and demonstrate their applicability to a variety of scientific problems.

\section{Mathematical Foundations of Neural Networks}
Neural networks are a type of machine learning model inspired by the human brain. A neural network consists of layers of interconnected nodes or neurons. Each connection between neurons has an associated weight, and each neuron has an associated bias. The output of each neuron is computed as a weighted sum of its inputs, passed through a non-linear activation function.

\subsection{Forward Propagation}
Forward propagation is the process of passing the input data through the network, layer by layer, until the output layer is reached. The output of each neuron in a layer is computed as follows:
\begin{equation}
z_i = \sum_{j=1}^{n} w_{ij} x_j + b_i
\end{equation}
\begin{equation}
a_i = f(z_i)
\end{equation}
where $x_j$ is the input to the neuron, $w_{ij}$ is the weight of the connection between the $i$-th neuron and the $j$-th input, $b_i$ is the bias of the $i$-th neuron, $z_i$ is the weighted sum of the inputs, $a_i$ is the output of the neuron, and $f$ is the activation function.

\subsection{Backward Propagation}
Backward propagation is the process of updating the weights and biases of the network to minimize the difference between the predicted output and the actual target values. The error for a single output neuron is given by:
\begin{equation}
E = \frac{1}{2}(t - a)^2
\end{equation}
where $t$ is the target value and $a$ is the output of the neuron.

\section{Physics Informed Neural Networks (PINNs)}
Physics Informed Neural Networks (PINNs) are a type of neural network that leverages the power of deep learning and the knowledge of the underlying physics of a problem to obtain more accurate and physically consistent solutions. PINNs are designed to satisfy the governing differential equations of a physical system, in addition to fitting the available data.

\subsection{Formulation of PINNs}
% Consider a physical system described by a partial differential equation (PDE) of the form:
% \begin{equation}
% \mathcal{L}(u; x) = 0, \quad x \in \Omega
% \end{equation}
% where $\mathcal{L}$ is a differential operator, $u$ is the unknown function to be determined, and $\Omega$ is the domain of interest.

% The idea of PINNs is to approximate the solution $u$ by a neural network $u_{NN}$ with parameters $\theta$, and to minimize the loss function that consists of two terms:
% \begin{equation}
% \mathcal{L}(\theta) = \mathcal{L}_{data}(\theta) + \lambda \mathcal{L}_{PDE}(\theta)
% \end{equation}
% where $\mathcal{L}_{data}$ is the data loss that measures the difference between the predicted values of $u_{NN}$ and the available data, $\mathcal{L}_{PDE}$ is the PDE loss that measures the difference between the predicted values of $\mathcal{L}(u_{NN}; x)$ and the true values (which are zero), and $\lambda$ is a hyperparameter that balances the two terms. The PDE loss is also often termed as the 'Physics Loss' in a lot of current literature.

% The Physics-Informed Neural Networks (PINNs) approach has been introduced as a means to integrate the structure and knowledge embedded in physical laws (often represented as partial differential equations, or PDEs) into the flexibility and capability of neural networks.

For a given physical system, let's consider it to be governed by a PDE of the form:
\begin{equation}
\mathcal{L}(u; x) = 0, \quad x \in \Omega
\end{equation}
Here, $\mathcal{L}$ denotes the differential operator, $u$ represents the unknown function we aim to solve for, and $\Omega$ stands for the domain in which the PDE is defined.

In the realm of PINNs, rather than seeking an analytical solution for $u$, we aim to approximate this solution using a neural network, which we'll denote as $u_{NN}$ with associated parameters $\theta$. The architecture of the neural network can be chosen based on the problem's complexity, with deeper networks being used for more intricate problems.

To train the neural network to be consistent with the physical system, a composite loss function is formulated as:
\begin{equation}
\mathcal{L}(\theta) = \mathcal{L}_{data}(\theta) + \lambda \mathcal{L}_{PDE}(\theta)
\end{equation}
The term $\mathcal{L}_{data}$ denotes the data-driven loss$^1$, quantifying the discrepancies between the neural network's predictions, $u_{NN}$, and any available observational data. On the other hand, $\mathcal{L}_{PDE}$, often referred to as the 'Physics Loss' in literature, encapsulates the error between the neural network's output when plugged into the PDE, i.e., $\mathcal{L}(u_{NN}; x)$, and the known ground truth (which is zero, based on the PDE definition). The hyperparameter $\lambda$ plays a pivotal role in weighting the importance between data fidelity and physical consistency.


To begin, let's reiterate our objective: we are looking to train a neural network to minimize a loss function $\mathcal{L}(\theta)$, which encapsulates both data fidelity and physical consistency. The optimization of this loss function can be represented as:

\begin{equation}
\theta^* = \underset{\theta}{\text{argmin}}  \mathcal{L}(\theta)
\end{equation}

Where $\theta^*$ is the optimal set of parameters for our neural network.\\

\textbf{Optimization:}
A plethora of optimization techniques can be employed, with gradient-based methods being particularly popular due to their efficiency in high-dimensional spaces. The backpropagation algorithm, facilitated by automatic differentiation, computes gradients of the loss function with respect to the neural network parameters. Common gradient-based optimizers include:
\begin{enumerate}
    \item Stochastic Gradient Descent (SGD)
    \item Adam
    \item RMSprop
\end{enumerate}

In the context of PINNs:
\begin{equation}
\theta_{t+1} = \theta_{t} - \eta \nabla_\theta \mathcal{L}(\theta_t)
\end{equation}

Where $\eta$ is the learning rate, and $\nabla_\theta \mathcal{L}(\theta_t)$ is the gradient of the loss with respect to parameters at iteration $t$.

The primary advantage of PINNs lies in their ability to generalize well even with sparse or noisy data, leveraging the embedded physical knowledge to guide the learning process. The method has gained traction due to its unique combination of data-driven learning with physics-based constraints.

\section{Applications of PINNs}
PINNs have been successfully applied to a wide range of problems in science and engineering, including fluid dynamics, heat transfer, and solid mechanics. In this section, we will present a few representative examples.

\subsection{Fluid Dynamics}
Consider the Navier-Stokes equations, which describe the motion of a fluid:
\begin{equation}
\rho \left(\frac{\partial u}{\partial t} + u \cdot \nabla u\right) = -\nabla p + \mu \nabla^2 u + f
\end{equation}
\begin{equation}
\nabla \cdot u = 0
\end{equation}
where $u$ is the velocity field, $p$ is the pressure, $\rho$ is the density, $\mu$ is the dynamic viscosity, and $f$ is the body force.

We implement the PINN solution to the 2D Navier-Stokes problem using the code and plot the pressure field \href{https://github.com/ComputationalDomain/PINNs/blob/main/Cylinder-Wake/NS_PINNS.py}{here}. \\

\begin{figure}[htp]
    \centering
    \includegraphics[width=8cm]{p field.png}
    \caption{PINN approximation to the pressure field for Navier Stokes Equation}
    \label{fig:Song}
\end{figure}

The neural network is trained to approximate the solution of these equations by using:
\begin{align*}
\psi(x,y,t) & : \text{Stream function from the neural network.} \\
p(x,y,t) & : \text{Pressure from the neural network.}
\end{align*}

The velocity fields $\Bar{u}$ and $\Bar{v}$ are derived from the stream function:

\begin{align*}
u &= \frac{\partial \psi}{\partial y} \\
v &= -\frac{\partial \psi}{\partial x}
\end{align*}

The residual (or physics-informed) loss components, represented as $\Bar{f}$ and $\Bar{g}$ in the code, ensure that the network's outputs satisfy the Navier-Stokes equations. They are given by:

\begin{align*}
f &= u_t + u u_x + v u_y + p_x - \nu (u_{xx} + u_{yy}) \\
g &= v_t + u v_x + v v_y + p_y - \nu (v_{xx} + v_{yy})
\end{align*}

Where subscripts denote partial derivatives. The total loss function that the neural network minimizes is a combination of the data mismatch (for $\Bar{u}$ and $\Bar{v}$) and the residuals (for $\Bar{f}$ and $\Bar{g}$).
\begin{equation*}
    \mathcal{L}(\theta) = \mathcal{L}_{u}(\theta) + \mathcal{L}_{v}(\theta) + \mathcal{L}_{f}(\theta) + \mathcal{L}_{g}(\theta)    
\end{equation*}


Where:
\begin{align*}
\mathcal{L}{u} & : \text{MSE loss between predicted and true values of } u. \\
\mathcal{L}{v} & : \text{MSE loss between predicted and true values of } v. \\
\mathcal{L}{f} & : \text{MSE loss between } f \text{ and 0 (since it should ideally be 0).} \\
\mathcal{L}{g} & : \text{MSE loss between } g \text{ and 0.}
\end{align*}

PINNs can be used to solve these equations by minimizing a loss function that consists of three terms: the data loss, the PDE loss, and the divergence-free loss, which ensures that the predicted velocity field satisfies the continuity equation.

\subsection{Heat Diffusion}

We aim to solve the 1D heat equation using Physics-Informed Neural Networks. The mathematical model we are considering is:

The 1D heat equation is given by:
\begin{equation}
    \frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}
\end{equation}\\

\textbf{Initial and Boundary Conditions}

The initial condition is:
\begin{equation}
    u(x, 0) = \sin(\pi x)
\end{equation}
And the boundary conditions are:
\begin{align}
    u(0, t) &= 0 \\
    u(1, t) &= 0
\end{align}\\

\textbf{Neural Network Architecture}
The neural network used is a simple feed-forward network with one hidden layer. The network takes two inputs ($x$ and $t$) and outputs the predicted solution $u(x,t)$. Activation function used is the hyperbolic tangent (tanh).\\

\textbf{Loss Function}
The loss function is constructed from three parts:
\begin{itemize}
    \item The PDE loss: This is the residual of the heat equation, given by:
    \begin{equation}
        L_{\text{PDE}} = \left(\frac{\partial u}{\partial t} - \frac{\partial^2 u}{\partial x^2}\right)^2
    \end{equation}
    
    \item The boundary conditions loss: Enforcing the values of $u$ at the boundaries to be zero.
    \begin{equation}
        L_{\text{BC}} = u^2(0,t) + u^2(1,t)
    \end{equation}
    
    \item The initial condition loss: Enforcing the neural network to match the given initial condition.
    \begin{equation}
        L_{\text{IC}} = \left(u(x,0) - \sin(\pi x)\right)^2
    \end{equation}
\end{itemize}
The total loss function is the sum of the three losses:
\begin{equation}
    L_{\text{total}} = L_{\text{PDE}} + L_{\text{BC}} + L_{\text{IC}}
\end{equation}\\

\textbf{Training Procedure}
Random samples from the domain are taken, and the loss function is minimized using the Adam optimizer. After training, the solution is compared to the analytical solution of the heat equation.

\textbf{Results}
The final plot shows the comparison of the neural network prediction at various training steps against the analytical solution. The analytical solution at a given time $t$ is:
\begin{equation}
    u_{\text{true}}(x,t) = e^{-\pi^2 t} \sin(\pi x)
\end{equation} 

\begin{figure}[htp]
    \centering
    \includegraphics[width=8cm]{1d heat 15000 deal.png}
    \caption{PINN approximation and exact solution to the 1d heat diffusion equation}
    \label{fig:Song}
\end{figure}
The convergence of the neural network solution towards the analytical solution is evident with increasing training steps.



\newpage
\section{Conclusion}
Physics Informed Neural Networks (PINNs) provide a novel approach for solving complex problems in science and engineering that are challenging for traditional solvers. By incorporating the knowledge of the underlying physics into the neural network, PINNs can obtain more accurate and physically consistent solutions. This report provides a comprehensive overview of the mathematical foundations of neural networks, explains the concept of PINNs, and demonstrates their applicability to a variety of scientific problems.

\section{References}
\begin{enumerate}
    \item Raissi, M., Perdikaris, P., \& Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.
\end{enumerate}



\end{document}
