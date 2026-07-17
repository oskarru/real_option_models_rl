# Reinforcement Learning in Real Option Models - Summary

## Introduction
Optimal stopping problems are concepts in the field of stochastic processes, in which the main objective is to determine the best time to take a particular action in order to maximize expected rewards or minimize costs under uncertainty. 

In practice, we can imagine an investor who invests their capital in a risky asset, such as the price of a particular stock on the stock market. The price of such an asset constantly fluctuates over time, driven by a random stochastic process (typically GBM). Such an investor would like to know when to sell their shares in order to maximize expected utility of his portfolio. This is an example of the optimal stopping problem.

Our main interest lies in the application of optimal stopping problems to real options, where the timing of an action in an uncertain environment plays a significant role. These approaches require knowledge of the underlying dynamics governing the system or access to critical parameters, about which we have limited knowledge in the real world situations. As a result, our implementations of optimal stopping problems are not sufficiently complex to be used in practical applications.

For this reason, we use reinforcement learning (RL) methods that do not require knowledge of the full specifics of the system's dynamics. They can work around model uncertainty by learning optimal policies directly from simulated trajectories.


## Theory
### Assumption 4.4
Assume the initial policy $g_0$ satisfies the following conditions:

1\. $g_0\in{C}^1([0,\hat{x}_{g_0}])$ is strictly increasing on $[0,\hat{x}_{g_0}]$,

2\. $g_0(0)=e^{-(1+\frac{\kappa \rho}{\lambda})}$,  and

3\. $-\alpha_-\Big(\kappa+\frac{\lambda}{\rho}\log(g_0(x))+\frac{\lambda}{\rho}\Big)+\alpha_-H_\pi(x)-H_\pi^{\prime}(x)\cdot x\ge 0$ on $[0,\hat{x}_{g_0}]$.

Note that 3. implies that $\hat{x}_{g_0}\leq\hat{x}_{g_\lambda}$.


## Model-Based Numerical Analysis - Algorithm 1
Before we move on to the RL algorithm, we will present a numerical algorithm for solving the optimal boundary problem in order to better illustrate the underlying mathematical theory. This algorithm makes use of the full knowledge of the system, hence it is called model-based.

### Algorithm 1. Policy Iteration for Entropy-regularized Optimal Stopping (PI-$\lambda$-OS)
1\. Initialize $g_0(x)$ for $x\in[0,\infty)$ according to Assumption 4.4

2\. **for** $k=0,1,\dots,K-1$

3\. &nbsp;&nbsp;&nbsp;&nbsp; Find $u_k(x,y)$ a ${C}^1(\mathbb{R}_+\times[0,1])\cap{C}^2\left(\overline{\mathcal{E}(g_k)}\right)$ solution to the following equations:

$$
(\mathcal{L}_x-\rho) u + \Big(\pi(x)-\rho \kappa \Big)y - \lambda y \log y =0\quad \text{on} \quad\mathcal{E}(g_k),
$$

$$
-u_y = 0, \quad \text{on}\quad \mathcal{S}(g_k).
$$

4\. &nbsp;&nbsp;&nbsp;&nbsp; Update the strategy:

$$
g_{k+1}(x) =
\begin{cases}
& \max \Big\{y< g_k(x)\,\Big| \partial_{xy} u_k(x,y) = 0 \Big\} \,\,\text{if} \,\, \partial_{xy}^- u_k(x,g_k(x))<0, \\
& g_{k+1}(x) = g_k (x) \qquad \text{otherwise.}  
\end{cases}
$$

5\. **end for**


## Model-Free Implementation - Algorithm 4
As we mentioned earlier, in the real world we will rarely know all the parameters of the model and the studied environment, without which we are unable to analytically determine the solution $u_k$. In this situation, we assume that we have access to an environment simulator (see Algorithm 3).

### Algorithm 3. Simulator $\mathcal{G}$
1\. **Input:** Threshold function $g$, initial position $(x,y)$

2\. **Generate:** Sample path $(X^{x},Y^{y,\xi^{g}})$ under policy $\xi^{g}$ (defined in equation)

3\. **Return:**

$$
\qquad \qquad \int_0^\infty e^{-\rho t} \Big(\Big( \pi(X^{x}_t)-\rho \kappa\Big)Y^{y,\xi^{g}}_t - \lambda Y^{y,\xi^{g}}_t \log (Y^{y,\xi^{g}}_t)\Big) \text{d}t
$$

For each initial position $(x,y)$ and threshold function $g$ provided by the learner, the generator will return an instantaneous reward function associated with a random path $X^x$ and the corresponding control $Y^{y,\xi^g}$ (see line 3. in Algorithm 3.). It is worth noting that the learner does not know the expression of the instantaneous reward function nor the generator of the dynamics.

### Algorithm 4. Sample-based Policy Iteration for Exploratory Optimal Stopping (SPI-$\lambda$-OS)
1\. Initialize $g_0(x)$ according to Assumption 4.4. Specify a grid size $\delta_x$ for partitioning the $x$-axis and a grid size $\delta_y$ for partitioning the $y$-axis. Also, specify an upper bound $\bar x:=N\delta_x$.

2\. **for** $k=0,1,\dots,K-1$

3\. &nbsp;&nbsp;&nbsp;&nbsp; **for** $x\in\{0,\delta_x,2\delta_x,\dots,N\delta_x\}$ and $y\in \{0,\delta_y,2\delta_y,\dots,\lfloor1/\delta_y\rfloor \}$

4\. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **for** $m=1,\dots,M$

5\. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Acquire the simulator $u^{(m)}_k(x,y) = \mathcal{G}(x,y,g_k)$

6\. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **end for**

7\. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Calculate the approximated value function $\bar u_k(x,y) = \frac{\sum_{m=1}^M u^{(m)}_k(x,y)}{M}$

8\. &nbsp;&nbsp;&nbsp;&nbsp; **end for**

9\. &nbsp;&nbsp;&nbsp;&nbsp; Update the strategy:

$$
g_{k+1}(x) =
\begin{cases}
& \max \Big\{y< g_k(x)\,\Big| \partial_{xy} \bar u_k(x,y) = 0 \Big\} \,\,\text{if} \,\, \partial_{xy}^- \bar u_k(x,g_k(x))<0, \\
& g_{k+1}(x) = g_k (x) \qquad \text{otherwise}  
\end{cases}
$$

10\. end for

By interacting with the generator, the learner approximates the value function by acquiring instantaneous rewards along multiple trajectories and take the average (see line 7. in Algorithm 4.). We require independent randomness across $M$ paths. Mathematically, it means that the state processes are driven by independent Brownian motions.  The learner then implements a sample-based version of the Policy Improvement step (see line 9. in Algorithm 4.). Note that the entire implementation avoids estimating model parameters. Instead, it iteratively updates the policy boundary. Therefore, this approach is referred to as a model-free implementation.

## References
Dianetti, J., Ferrari, G., & Xu, R. (2026). *Reinforcement Learning in Real Option Models*