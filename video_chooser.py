import pyglet

from widgets import WindowWidget, CenteredWidget, RelativeWidget, Button


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

# This script is for process 2:
# 1) Receive trajectories segment pairs σ1 and σ2 into a queue
# 2) Human chooses preference of one trajectory segment over another
# 3) Triple (σ1, σ2, μ) is generated. μ is a distribution over {1,2} indicating which segment the user preferred
#    If the human selects one segment as preferable, then μ puts all of its mass on that choice. If the human 
#    marks the segments as equally preferable, then μ is uniform. Finally, if the human marks the segments as 
#    incomparable, then the comparison is not included in the database.

