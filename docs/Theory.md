# Reinforcement Learning in Real Option Models - Summary

## Introduction
Optimal stopping problems are concepts in the field of stochastic processes, in which the main objective is to determine the best time to take a particular action in order to maximize expected rewards or minimize costs under uncertainty. 

In practice, we can imagine an investor who invests their capital in a risky asset, such as the price of a particular stock on the stock market. The price of such an asset constantly fluctuates over time, driven by a random stochastic process (typically GBM). Such an investor would like to know when to sell their shares in order to maximize expected utility of his portfolio. This is an example of the optimal stopping problem.

Our main interest lies in the application of optimal stopping problems to real options, where the timing of an action in an uncertain environment plays a significant role. These approaches require knowledge of the underlying dynamics governing the system or access to critical parameters, about which we have limited knowledge in the real world situations. As a result, our implementations of optimal stopping problems are not sufficiently complex to be used in practical applications.

For this reason, we use reinforcement learning (RL) methods that do not require knowledge of the full specifics of the system's dynamics. They can work around model uncertainty by learning optimal policies directly from simulated trajectories.


## Theory
The classic equation of wealth dynamics is Geometric Brownian Motion, expressed by the following formula

$$
\text{d} X_t^x=\mu X_t^x\,\text{d} t+\sigma X_t^x\,\text{d} W_t,\qquad X_0^x=x>0,
$$

where expected rate of return (drift) $\mu \in \mathbb{R}$ and standard deviation of returns $\sigma > 0$.

We consider the following classical real option/optimal stopping problem

$$
V(x)=\sup_{\tau\in\mathcal T}\mathbb{E}\left[\int_0^\tau e^{-\rho t}\bigl(\pi(X_t^x)-\rho\kappa\bigr)\,\text{d} t\right],
$$

where $\rho > \mu$ and running profit $\pi: \mathbb{R}_+ \rightarrow \mathbb{R}$ is expressed as

$$
\pi(x)=x^\theta, \qquad 0<\theta<1.
$$

### Entropy regularization
To encourage exploration by the agent, we must introduce a regularization term so the optimal strategy incorporates randomness.By punishing purely deterministic policies, the entropy term forces the agent to maintain a probability distribution over its actions. This prevents premature convergence to a suboptimal strategy and naturally drives the agent to continuously explore different states.

We introduce the extra state variable $y\in[0,1]$ describing the probability of continuation (i.e., the likelihood that the agent has not yet exercised the option up to the current time). For $\lambda > 0$, we define the updated singular control problem

$$
V^\lambda(x,y)=\sup_{\xi}\mathbb{E}\left[\int_0^\infty e^{-\rho t}\Bigl((x^\theta-\rho\kappa)Y_t^{y,\xi}-\lambda Y_t^{y,\xi}\log Y_t^{y,\xi}\Bigr)\,\text{d} t\right].
$$

To solve this stochastic control problem and find the optimal value function $V^\lambda(x,y)$, we rely on the Hamilton-Jacobi-Bellman (HJB) equation, which is the fundamental partial differential equation defining the necessary and sufficient conditions for optimality. In our case, the HJB equation takes the following form

$$
\max\left\{
		(\mathcal L_x-\rho)V^\lambda(x,y)
		+
		(x^\theta-\rho\kappa)y
		-\lambda y\log y,\,
		-\partial_y V^\lambda(x,y)
		\right\}=0,
$$

with boundary condition $V(x, 0)=0$, $x \in (0,\infty)$, where  $\mathcal{L}_x \phi (x) := \mu x \phi_x (x) + \frac{1}{2} \sigma^2 {x^2} \phi_{xx} (x)$.

### The Optimal Decision Boundary
Before presenting the formal analytical solution, it is crucial to understand the role of the threshold function $g(x)$. In optimal stopping problems, $g(x)$ acts as a decision boundary (often called the free boundary) that splits the state space into two regions: the continuation region (where the agent should wait) and the stopping region (where the agent should take action). In our regularized setting, $g_\lambda(x)$ defines the optimal threshold for the continuation probability $y$.

### Theorem 3.1 (The solution to the entropy-regularized real option problem)
Let $(x,y) \in (0,\infty) \times [0,1]$ be given and fixed.
Introduce the nondecreasing function

$$
    g_{\lambda}(x) := \exp \left( \frac{-\frac{H_\pi'(x)}{\alpha_-}x+H_\pi(x)-\kappa - \frac{\lambda}{\rho}}{\frac{\lambda}{\rho}}\right) \wedge 1,
$$

such that $y^\lambda:= g_\lambda(0) = e^{-1-\frac{\kappa \rho}{\lambda}}$,
and define

$$
H_\pi(x) := \mathbb{E}\left[\int_0^\infty e^{-\rho t}\pi(X_t^x)\text{d} t\right],
$$

$$
A_{2}(y) :=\int_{g_\lambda(0)}^y  \frac{\kappa  +\frac{\lambda}{\rho}  \log (u)+ \frac{\lambda}{\rho}- H_\pi(g_\lambda^{-1}(u))}{(g_\lambda^{-1}(u))^{\alpha_-}} \text{d} u.
$$

Then, letting $F(x, y):=A_{2} (y) \,x^{\alpha_-} + H_\pi(x)y - \kappa y -\frac{\lambda}{\rho} y \log y$, the value of the entropy-regularized real option problem is given by 

$$
V^{\lambda}(x,y) =
\begin{cases}
H_\pi(x)y - \kappa y - \frac{\lambda}{\rho} y \log y, & y<y^\lambda, \\
F(x, y), & y\leq g_\lambda (x), \ y \geq y^\lambda, \\
F(x, b_\lambda^{-1} (x) ), & y> g_\lambda (x), \ y \geq y^\lambda, 
\end{cases}
$$

and the reflection policy 

$$
\xi_{t}^\lambda := 
\sup\limits_{0 \leq s \leq t} \big( y - g_{\lambda} (X_s^x) \big)^{+}, \quad t>0, \quad \xi_{0^-}^\lambda =0,
$$ 

is optimal for the initial condition $(x,y) \in (0,\infty) \times [0,1]$.

### Reinforcement Learning and Policy Initialization
While Theorem 3.1 provides the exact mathematical solution, deriving it analytically is often impossible for more complex environments. Therefore, we utilize Reinforcement Learning (RL) algorithms to approximate the optimal boundary numerically. In this approach, the learning agent must start with a prior guess of the boundary - an initial policy $g_0(x)$ - and iteratively improve it by interacting with the environment and collecting simulated data.

### Assumption 4.4
Assume the initial policy $g_0$ satisfies the following conditions:

1\. $g_0\in{C}^1([0,\hat{x}_{g_0}])$ is strictly increasing on $[0,\hat{x}_{g_0}]$,

2\. $g_0(0)=e^{-(1+\frac{\kappa \rho}{\lambda})}$,  and

3\. $-\alpha_-\Big(\kappa+\frac{\lambda}{\rho}\log(g_0(x))+\frac{\lambda}{\rho}\Big)+\alpha_-H_\pi(x)-H_\pi^{\prime}(x)\cdot x\ge 0$ on $[0,\hat{x}_{g_0}]$.

Note that 3. implies that $\hat{x}_{g_0}\leq\hat{x}_{g_\lambda}$.

### Initialization functions that satisfy Assumption 4.4
For linear initialization, we take:

$$
    g_0(x) = \min\left\{e^{-(1+\frac{\kappa\rho}{\lambda})} +2 (1-e^{-(1+\frac{\kappa\rho}{\lambda})})\Big(\frac{- \alpha_- \big(\kappa  +\frac{\lambda}{\rho}\big)}{\theta-\alpha_-}\Big)^{\theta}x, 1\right\},
$$

which interpolates $(x_0,y_0)$ and $(x_1,y_1)$ with
$x_0=0$, $y_0=e^{-(1+\frac{\kappa\rho}{\lambda})}$, $x_1 =  \frac{1}{2}\Big(\frac{- \alpha_- \big(\kappa  +\frac{\lambda}{\rho}\big)}{\theta-\alpha_-}\Big)^{-\theta}$ and $y_1=1$. 

For exponential initialization, we can take 

$$
    g_0(x) = \min\left\{\exp\Big(\frac{\rho (\zeta-\alpha_-) (x)^{\zeta}}{- \alpha_-\lambda} - \frac{\rho}{\lambda}( \kappa+  \frac{\lambda}{\rho} )\Big), 1\right\},
$$

for some $\zeta\in(\theta,1)$.

Furthermore, based on Theorem 3.1 we have defined an explicit form of the boundary for the power function

$$
g_\lambda(x)
		=
		\exp\left(
		\frac{\rho}{\lambda}
		\left[
		\frac{x^\theta}{P}\left(1-\frac{\theta}{\alpha_-}\right)-\kappa
		\right]
		-1
		\right)\wedge 1,
$$

where

$$
P=\rho-\theta\mu-\frac{\sigma^2}{2}\theta(\theta-1),
$$

$$
\alpha_\pm=
		\frac{ -\left(\mu-\frac{\sigma^2}{2}\right)\pm
			\sqrt{\left(\mu-\frac{\sigma^2}{2}\right)^2+2\rho\sigma^2}}
		{\sigma^2}.
$$

(This function will serve as out $g_{\text{true}}$ and analytical initialization for testing Algorithm 1 and 4.)

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

Algorithm 1 follows a classical Policy Iteration framework consisting of two main phases: Policy Evaluation (Step 3) and Policy Improvement (Step 4).

**Policy Evaluation**: The threshold function $g_k(x)$ splits the state space into two distinct regions: the continuation/exploration region $\mathcal{E}(g_k)$ (where $y < g_k(x)$) and the stopping region $\mathcal{S}(g_k)$ (where $y \geq g_k(x)$). In Step 3, the algorithm evaluates the current policy by analytically solving the continuous-time Hamilton-Jacobi-Bellman (HJB) equation. In the continuation region $\mathcal{E}(g_k)$, the value function $u_k$ satisfies the complex partial differential equation (PDE) driven by the infinitesimal generator $\mathcal{L}_x$. In the stopping region $\mathcal{S}(g_k)$, the optimal action is to stop, which enforces the boundary condition $-u_y = 0$.

**Policy Improvement**: In Step 4, the algorithm attempts to update the threshold $g_k(x)$ by examining the marginal value of continuation. If the left-sided mixed derivative $\partial_{xy}^- u_k(x,g_k(x))$ is strictly negative, it indicates that the current threshold is placed too high, and the marginal benefit of continuing is decreasing. The algorithm then lowers the boundary by finding a new optimal point $y < g_k(x)$ where the mixed derivative equals zero (indicating a local maximum of the continuation value).


## Model-Free Implementation - Algorithm 4
Algorithm 1 represents a classical Stochastic Control approach (Dynamic Programming): it requires full, explicit knowledge of the system's governing equations (such as the exact physical laws, transition probabilities, and the HJB formulation) to compute the solution mathematically. It does not "learn"; it calculates. That's why we need to introduce the model-free approach. In this situation, we assume that we have access to an environment simulator.

### Algorithm 3. Simulator $\mathcal{G}$
1\. **Input:** Threshold function $g$, initial position $(x,y)$

2\. **Generate:** Sample path $(X^{x},Y^{y,\xi^{g}})$ under policy $\xi^{g}$ (defined in equation)

3\. **Return:**

$$
\qquad \qquad \int_0^\infty e^{-\rho t} \Big(\Big( \pi(X^{x}_t)-\rho \kappa\Big)Y^{y,\xi^{g}}_t - \lambda Y^{y,\xi^{g}}_t \log (Y^{y,\xi^{g}}_t)\Big) \text{d}t
$$

Algorithm 3 serves as the Environment. For a given starting state $(x,y)$ and a specific threshold policy $g$, the simulator generates a single random Monte Carlo sample path (governed by the underlying Geometric Brownian Motion dynamics). It then calculates and returns the total discounted cumulative reward for that specific trajectory based on the regularized optimal stopping formula. The learning agent relies entirely on these returned rewards, without needing any prior knowledge of the underlying stochastic differential equations or the explicit HJB formulation.

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

10\. **end for**

Algorithm 4 is the model-free counterpart to Algorithm 1. It approximates the optimal boundary using empirical data generated by the simulator (Algorithm 3).

**Empirical Policy Evaluation (Steps 3-8)**: Instead of solving the HJB equation across the entire continuous space, the algorithm discretizes the state space into a grid. For each coordinate on the grid, the agent queries the simulator to generate $M$ independent Monte Carlo sample paths (Steps 4-6). By averaging the cumulative rewards from these paths (Step 7), the agent calculates an empirical estimate of the value function $\bar u_k(x,y)$ according to the Law of Large Numbers.

**Strategy Update (Step 9)**: The improvement step functions identically to Algorithm 1. However, instead of finding the root of an analytical derivative, the algorithm numerically investigates the estimated function $\bar u_k(x,y)$ to adjust the threshold $g_{k+1}(x)$ based on the approximated gradients.

Algorithm 4 operates strictly under the Reinforcement Learning (RL). The RL agent does not have access to the underlying stochastic differential equations governing the asset's price. Instead, it must discover the optimal strategy through trial-and-error interaction. By continuously querying the environment simulator (Algorithm 3), collecting random sample paths, and observing the resulting delayed rewards, the agent builds an empirical estimation of the value function based solely on its simulated experiences.

## References
Dianetti, J., Ferrari, G., & Xu, R. (2026). [*Reinforcement Learning in Real Option Models*](https://arxiv.org/abs/2602.15643)