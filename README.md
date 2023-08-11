# RL_Dual_Sourcing
Applying Actor-Critic Method to the Dual Sourcing Inventory problem. Jun-Aug 2023

## Summary of our work
In this paper, we analyze the performance of Advantage Actor-Acritic(A2C) algorithm against
the traditional Tailored Base-Surge(TBS) policy in the dual-sourcing problem through three experi-
ments. First, we test the algorithm in a small-scale environment with different demand distributions.
The compact state-action space enables us to effectively experiment different exploration strategies
while controlling the learning rate. Next, we simulate a demand shock by altering the expected value
and variance of the demand in the middle of an episode. We assess the algorithm’s resilience to sud-
den changes and measure its adaptability by considering its time-to-convergence and average reward
levels. Lastly, we apply A2C to a historical demand dataset from a global manufacturing company,
evaluating its behavior in a real-market setting.

Our results reveal several interesting insights. In the first experiment, while both the A2C agents
and TBS policy yield costs roughly double the theoretical optimum, A2C’s performance declines
with increased demand volatility. In the simulated shock, A2C demonstrates adaptability by quickly
converging back to a reasonable stock level, while TBS struggles to recover. Real-world data shows
that overall, A2C exhibits a riskier approach with quick adaptability, while TBS’s conservative strat-
egy ensures stability but often leads to overstocking. Notably, both policies perform comparably in
average daily cost, even though TBS had a much better performance in volatile environments in our
first experiment. This result illustrates potential weaknesses in evaluating the performance of A2C
Agents solely based on simulated environments. Our findings offer valuable insights into applying
reinforcement learning techniques in supply chain optimization and their adaptability to real-world
scenarios

## Problem Formulation
In the dual-sourcing problem, the inventory can be replenished at unit cost \(c_r\) from a regular supplier \(R\) with lead time \(L_r\) or/and from an express source \(E\) with lead time \(L_e\) at premium unit cost \(c_e\). In the beginning of any timestamp \(t\), two order quantities, \(q^r_t\) and \(q^e_t\), must be decided after observing the last inventory level on hand, \(I_{t-1}\), and outstanding receipts from regular and express suppliers, \(\mathbf{Q}^{r}_{t-1}= (q^r_{t-L_r},q^r_{t-L_r+1}, \dots,  q^r_{t-1})\) and \(\mathbf{Q}^{e}_{t-1}= (q^e_{t-L_e},q^e_{t-L_e+1}, \dots,  q^e_{t-1})\).

After the order decision, orders \(q^r_{t-L_r}\) and \(q^e_{t-L_e}\) are received and added to the on-hand inventory. Then, the unknown demand \(D_t\) is realized and subtracted from the on-hand inventory. Excess demand is fully backlogged so that the inventory and outstanding receipts evolve as:
$$
I_{t+1} = I_t + q^r_{t-L_r} + q^e_{t-L_e} - D_{t}
$$
Finally, outstanding pipeline vectors are updated as:
$$
\mathbf{Q}^{r}_{t} = (q^r_{t-L_r+1},q^r_{t-L_r+2}, \dots,  q^r_{t})
$$ 
and 
$$
\mathbf{Q}^{e}_{t} = (q^e_{t-L_e+1},q^e_{t-L_e+2}, \dots,  q^e_{t})
$$

The dual-sourcing problem can be modeled as a Discrete Markov Decision Process with states represented by the on-hand inventory level and the pipeline vectors: \(\textbf{S}_t = (I_{t-1}, \mathbf{Q}^{r}_{t-1},\mathbf{Q}^{e}_{t-1})\). The action taken in period \(t\) is now two-dimensional: \(\textbf{a}_t = (q^r_t, q^e_t)\) consisting of the ordered quantities from the regular and expedited sources. To decide what new orders at \(t\) should be, we need a policy that maps states to actions, which is defined as \(\pi\): \(\{f_{t}^{\pi}, t \geq 0\}\), with \((q^r_t, q^e_t) = f^{\pi}_t(\textbf{q}^r_t, \textbf{q}^e_t, I_t)\).

