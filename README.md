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
 In the dual-sourcing problem, the inventory can be replenished at unit cost c_r from a regular supplier $R$ with lead time $L_r$ or/and from an express source $E$ with lead time $L_e$ at premium unit cost $c_e$. In the beginning of any timestamp $t$, two order quantities, $q^r_t$ and $q^e_t$, and must be decided after observing the last inventory level on hand, $I_{t-1}$, and outstanding receipts from regular and express suppliers, $\mathbf{Q}^{r}_{t-1}= \left( q^r_{t-L_r},q^r_{t-L_r+1}, \dots,  q^r_{t-1} \right) $ and $\mathbf{Q}^{e}_{t-1}= \left( q^e_{t-L_e},q^e_{t-L_e+1}, \dots,  q^e_{t-1} \right) $. 
