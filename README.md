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

## Brief Intro 
One common practice in supply chain management is Dual Sourcing, which is also known as dual supply or dual vendor strategy. This strategy involves establishing relationships with two or more suppliers for a particular product, component, or raw material, thereby providing an alternative source of supply in case one supplier encounters disruptions or fails to meet the required business objectives. Considering the variability in both price and delivery time across different suppliers, the enterprise need to find an optimal strategy of managing the multiple suppliers in order to maximize their own profitability.

Under the dual sourcing problem, inventory can be replenished through two avenues: a slow and economical source, and a faster but more expensive source. This parallels the dual-mode problem, where inventory is restocked from a single supplier utilizing two complementary modes of transportation. In this paper, we call the former the regular supplier and the later express supplier, and their different capabilities are reflected in parameters regular/express lead time and regular/express costs. 

## Problem Formulation
![Hi Image](./report%2Bposter/all%20figures/rl_dual_sourcing_problem_formulation.png)
