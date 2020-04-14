# Overall processes:
# A trajectory segment is a sequence of observations and actions, σ = ((o0,a0),(o1,a1),...,(ok−1,ak−1))∈(O×A)k. 
# Write σ1 > σ2 to indicate that the human preferred trajectory segment σ1 to trajectory segment σ2.
#
# These networks are updated by three processes:
# 1. The policy π interacts with the environment to produce a set of trajectories {τ1,...,τi}. The parameters of π 
#    are updated by a traditional reinforcement learning algorithm, in order to maximize the sum of the predicted
#    rewards rt = r(ot, at).
# 2. We select pairs of segments (σ1,σ2) from the trajectories {τ1,...,τi} produced in step 1, and send them to a
#    human for comparison.
# 3. The parameters of the mapping r are optimized via supervised learning to fit the comparisons collected from
#    the human so far.

# This script is for process 3:
# 1) Receive evaluated trajectory tuples (σ1, σ2, u) from process 2 into a queue
# 2) Agent retrieves item from queue and stores them locally into a circular buffer to hold the last N comparisons
# 3) Triple (σ1, σ2, μ) is evaluated by agent, improving neural net model that predicts reward.
# 4) Parameters for the model are saved into a variable that can be accessed by process 1.
#
#

