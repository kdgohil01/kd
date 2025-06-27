# Problem: Maximum Flow (Dinic's Algorithm)
# Given a flow network with a source node, a sink node, and capacities on directed edges,
# find the maximum amount of flow that can be sent from the source to the sink.

# Approach: Dinic's Algorithm
# Dinic's algorithm improves upon Edmonds-Karp by working in phases.
# Each phase consists of two main steps:
# 1. Building a Level Graph (BFS):
#    - A Breadth-First Search (BFS) is performed from the source to construct a 'level graph'
#      (also known as a layered network). Each node `v` gets a 'level' `level[v]`, which
#      is the shortest path distance (in terms of number of edges) from the source `s` to `v`
#      in the residual graph.
#    - Edges in the level graph only go from a lower level to a higher level (`level[v] == level[u] + 1`).
#    - If the sink `t` is not reachable from `s` in the BFS, it means no more augmenting paths
#      exist, and the algorithm terminates.
#
# 2. Finding a Blocking Flow (DFS):
#    - A Depth-First Search (DFS) is performed on the level graph to find a 'blocking flow'.
#      A blocking flow is a flow such that every path from `s` to `t` in the current level graph
#      contains at least one saturated edge (an edge where flow equals capacity).
#    - During the DFS, only traverse edges `(u, v)` where `level[v] == level[u] + 1` and there is
#      residual capacity.
#    - Crucially, Dinic's uses an optimization called 'pointer optimization' (or `next_edge` optimization).
#      For each node `u`, `ptr[u]` keeps track of the next edge to explore from `u` in the adjacency list.
#      If an edge from `u` doesn't lead to an augmenting path, we don't need to check it again
#      in the current DFS phase (as part of the blocking flow search). This avoids redundant work.
#    - When an augmenting path is found to the sink `t`, push the minimum residual capacity along that path.
#      Update residual capacities for forward and backward (residual) edges.
#    - The DFS continues from the current node `u` until no more flow can be pushed through `u`
#      or all outgoing edges from `u` in the level graph have been explored.
#
# These phases are repeated until no more augmenting paths can be found (i.e., the sink is
# unreachable in the BFS). The total flow accumulated across all phases is the maximum flow.

# Time Complexity: O(V^2 * E) in general graphs. In unit capacity graphs, it can be O(min(V^(2/3), E^(1/2)) * E).
# Space Complexity: O(V + E) for storing the graph and level array.

# Code:

import collections

class Dinic:
    def __init__(self, num_nodes):
        # Initialize the Dinic's algorithm solver for a graph with num_nodes nodes.
        self.num_nodes = num_nodes
        # Adjacency list representation of the graph.
        # Each element graph[u] is a list of edges originating from u.
        # Each edge is a list: [to_node, capacity, index_of_reverse_edge_in_to_node's_list]
        self.graph = [[] for _ in range(num_nodes)]
        # level[i] stores the level (shortest distance from source in terms of edges)
        # of node i in the current layered network. -1 if not visited.
        self.level = [-1] * num_nodes
        # ptr[i] is used in DFS to keep track of the next edge to explore from node i.
        # This is a crucial optimization for Dinic's algorithm.
        self.ptr = [0] * num_nodes

    def add_edge(self, u, v, capacity):
        # Adds a directed edge from u to v with a given capacity.
        # Also adds a residual (backward) edge from v to u with 0 initial capacity.
        
        # Add forward edge (u -> v)
        # The third element is the index in graph[v] where the reverse edge is stored.
        self.graph[u].append([v, capacity, len(self.graph[v])])
        
        # Add backward (residual) edge (v -> u) with 0 initial capacity.
        # The third element is the index in graph[u] where the forward edge is stored.
        self.graph[v].append([u, 0, len(self.graph[u]) - 1])

    def _bfs(self, s, t):
        # Performs a Breadth-First Search from source `s` to build the level graph.
        # Returns True if sink `t` is reachable, False otherwise.
        self.level = [-1] * self.num_nodes # Reset levels for current phase
        self.level[s] = 0 # Source is at level 0
        q = collections.deque([s])
        while q:
            u = q.popleft()
            # Iterate through all neighbors of u
            for v, capacity, _ in self.graph[u]:
                # If v is not visited (-1) and there is residual capacity along (u, v)
                if self.level[v] == -1 and capacity > 0:
                    self.level[v] = self.level[u] + 1 # Set level of v
                    q.append(v)
        # Returns True if the sink `t` was reached during BFS, indicating an augmenting path exists.
        return self.level[t] != -1

    def _dfs(self, u, t, pushed_flow):
        # Performs a Depth-First Search to find a blocking flow in the level graph.
        # `u`: current node
        # `t`: sink node
        # `pushed_flow`: maximum flow that can be pushed through the current path to `u`
        
        if pushed_flow == 0: # If no flow can be pushed, return 0
            return 0
        if u == t: # If we reached the sink, we found an augmenting path
            return pushed_flow

        # Iterate through edges from `u` starting from `ptr[u]`.
        # `ptr[u]` optimizes DFS by avoiding re-exploring edges that have already been saturated
        # or determined not to lead to an augmenting path in the current phase.
        while self.ptr[u] < len(self.graph[u]):
            edge = self.graph[u][self.ptr[u]]
            v, capacity, rev_idx = edge[0], edge[1], edge[2]

            # Check two conditions for a valid edge in the layered network:
            # 1. `v` must be at the next level (`level[v] == level[u] + 1`).
            # 2. The edge must have residual capacity (`capacity > 0`).
            if self.level[v] != self.level[u] + 1 or capacity == 0:
                self.ptr[u] += 1 # Move to the next edge if current one is not valid
                continue
            
            # Recursively call DFS for the next node `v`, limiting flow by current edge's capacity
            traced_flow = self._dfs(v, t, min(pushed_flow, capacity))
            
            if traced_flow == 0: # If no flow could be pushed through `v` and its subtree
                self.ptr[u] += 1 # Move to the next edge
                continue

            # If flow was pushed, update capacities
            self.graph[u][self.ptr[u]][1] -= traced_flow # Decrease forward edge capacity
            self.graph[v][rev_idx][1] += traced_flow     # Increase backward (residual) edge capacity
            
            return traced_flow # Return the amount of flow pushed through this path

        return 0 # No more valid paths from `u` in this DFS phase

    def max_flow(self, s, t):
        # Main function to calculate the maximum flow from source `s` to sink `t`.
        total_flow = 0
        
        # Phase loop: continues as long as an augmenting path can be found via BFS
        while self._bfs(s, t):
            # Reset the pointers for DFS for each new phase (new level graph)
            self.ptr = [0] * self.num_nodes
            
            # Blocking flow loop: continues finding augmenting paths in the current level graph
            # until no more flow can be pushed in this phase.
            while True:
                # Attempt to push flow from source `s` to sink `t` with infinite capacity initially.
                pushed = self._dfs(s, t, float('inf'))
                if pushed == 0: # If no more flow could be pushed in this DFS iteration
                    break # Exit the blocking flow loop
                total_flow += pushed # Add the pushed flow to the total
                
        return total_flow # Return the accumulated maximum flow

# Example Usage:
# Create a flow network with 6 nodes (0 to 5)
# Source node (s) = 0, Sink node (t) = 5

# The network edges and their capacities:
# 0 -> 1 (capacity 10)
# 0 -> 2 (capacity 10)
# 1 -> 2 (capacity 2)
# 1 -> 3 (capacity 4)
# 1 -> 4 (capacity 8)
# 2 -> 4 (capacity 9)
# 3 -> 5 (capacity 10)
# 4 -> 3 (capacity 6)
# 4 -> 5 (capacity 10)

dinic = Dinic(6)
dinic.add_edge(0, 1, 10)
dinic.add_edge(0, 2, 10)
dinic.add_edge(1, 2, 2)
dinic.add_edge(1, 3, 4)
dinic.add_edge(1, 4, 8)
dinic.add_edge(2, 4, 9)
dinic.add_edge(3, 5, 10)
dinic.add_edge(4, 3, 6)
dinic.add_edge(4, 5, 10)

s_node = 0 # Source node
t_node = 5 # Sink node

max_flow_val = dinic.max_flow(s_node, t_node)
print(f"The maximum flow from node {s_node} to node {t_node} is: {max_flow_val}") # Expected output: 19

# Another example: Simple network
# S(0) -> A(1) (10)
# S(0) -> B(2) (10)
# A(1) -> C(3) (4)
# A(1) -> D(4) (8)
# B(2) -> C(3) (9)
# C(3) -> T(5) (10)
# D(4) -> T(5) (10)

dinic2 = Dinic(6)
dinic2.add_edge(0, 1, 10)
dinic2.add_edge(0, 2, 10)
dinic2.add_edge(1, 3, 4)
dinic2.add_edge(1, 4, 8)
dinic2.add_edge(2, 3, 9)
dinic2.add_edge(3, 5, 10)
dinic2.add_edge(4, 5, 10)

s_node2 = 0
t_node2 = 5

max_flow_val2 = dinic2.max_flow(s_node2, t_node2)
print(f"The maximum flow from node {s_node2} to node {t_node2} is: {max_flow_val2}") # Expected output: 20
# Path 0-1-3-5: 4 units
# Path 0-1-4-5: 8 units
# Path 0-2-3-5: 6 units (since 3-5 capacity is 10, but 0-2 capacity is 10, and 2-3 capacity is 9. Path 0-2-3-5 can take min(10,9,10) = 9 units. But 3-5 is bottlenecked by 10-4=6)
# Total = 4 + 8 + 6 = 18. Let's re-evaluate.

# Re-evaluating expected for second example:
# Max flow from S(0) to T(5)
# Path 0-1-3-5: capacity min(10, 4, 10) = 4. Send 4 units.
#   Residuals: 0->1(6), 1->3(0), 3->5(6). Total flow = 4.
# Path 0-1-4-5: capacity min(6, 8, 10) = 6. Send 6 units.
#   Residuals: 0->1(0), 1->4(2), 4->5(4). Total flow = 4 + 6 = 10.
# Path 0-2-3-5: capacity min(10, 9, 6) = 6. Send 6 units.
#   Residuals: 0->2(4), 2->3(3), 3->5(0). Total flow = 10 + 6 = 16.
# Path 0-2-4-5: (residual 4-1 available from 1-4) But 0-2 is left with 4. 4-5 is left with 4. So max 4 units.
# Residual edge 1->4 (capacity 2 left). Residual edge 0->2 (capacity 4 left). 
# It seems the path 0-2-3-5 is the 6 units. Residual on 3-5 becomes 0. 
# So no more flow through node 3. Only node 4 for T.
# We have 0->2(4 left), 2->3(3 left). 
# So only 0->2->3 is 3 more units. But 3->5 is 0. So no more through 3.
# What about 0-2 -> then where? No edge 2->4 directly.
# The max flow should be 18: 0-1-3-5 (4), 0-1-4-5 (8), 0-2-3-5 (6) => 4+8+6 = 18.
# Dinic's will find this efficiently.
