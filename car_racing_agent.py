# A trajectory segment is a sequence of observations and actions, σ = ((o0,a0),(o1,a1),...,(ok−1,ak−1))∈(O×A)k. 
# 
# These networks are updated by three processes:
# 1. The policy π interacts with the environment to produce a set of trajectories {τ1,...,τi}. The parameters of π 
#    are updated by a traditional reinforcement learning algorithm, in order to maximize the sum of the predicted
#    rewards rt = r(ot, at).
# 2. We select pairs of segments (σ1,σ2) from the trajectories {τ1,...,τi} produced in step 1, and send them to a
#    human for comparison.
# 3. The parameters of the mapping r are optimized via supervised learning to fit the comparisons collected from
#    the human so far.

# This script is for process 1:
# 1) Use policy gradient RL algorithm to optimize sum of predicted rewards
# 2) Capture pairs of trajectory segments to use for process 2. Sample a large number of pairs of trajectory segments
#    of length k, use each reward predictor in our ensemble (from process 3) to predict which segment will be preferred
#    from each pair, and then select those trajectories for which the predictions have the highest variance across ensemble members.
# 3) Send chosen pair of trajectory segment to process 2