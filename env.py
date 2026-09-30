import numpy as np

class AllNodesDeadError(Exception):
    """Custom exception to indicate that all nodes in the environment are dead."""
    pass

class Environment:
    def __init__(self, adjacency_matrix, failure_rate, length_message, message_candidates,
                 rng, max_dead_nodes=None, p_loss_max=0.05, loss_concentration=4.0,
                 neighborhood_reward_weight=0.0):
        """Crash-prone nodes on a graph, exchanging messages over lossy links.

        Each round every alive node crashes with probability failure_rate
        (at most max_dead_nodes dead at once, and never all of them), and each
        received message is dropped with a congestion-dependent probability
        capped at p_loss_max. The reward of an alive node is minus its squared
        belief error about every node.
        """
        self.adjacency_matrix: np.ndarray = adjacency_matrix
        self.p_failure: float = failure_rate
        self.length_message: int = length_message
        self.message_candidates: np.ndarray = message_candidates

        # Credit assignment for the learned message channel. With the default 0.0
        # each node is rewarded only for its own belief accuracy — but a symbol a
        # node emits only ever improves its *neighbors'* beliefs, so the emitter is
        # never credited for speaking well and the Q-slots have no gradient to
        # follow. A positive weight adds the mean belief error of the nodes that
        # actually receive this node's message, making "being useful to speak to"
        # a term the message head can optimize.
        self.neighborhood_reward_weight: float = float(neighborhood_reward_weight)

        # Congestion-driven, node-wise packet loss.
        # Each step every node draws its own loss rate from a Beta kernel whose
        # mean scales with the fraction of in-neighbors that sent a real message
        # this step. p_loss_max caps the loss rate; loss_concentration controls
        # how tightly the drawn rate concentrates around that mean.
        self.p_loss_max: float = p_loss_max
        self.loss_concentration: float = loss_concentration

        self.rng = rng

        # Priming the first frame with random symbols used to draw
        # number_of_nodes x length_message values out of the *dynamics* stream at
        # every reset. Since length_message is exactly what the Q-slot ablation
        # changes (4N vs 2N), the two conditions consumed a different number of
        # draws per episode and their failure and packet-loss sequences drifted
        # apart -- 3x3 saw 175 deaths under Full against 133 under Ablation over
        # the same 5 episodes, diverging from step 7. The comparison was
        # therefore not controlled.
        #
        # The priming noise now comes from its own streams, so the dynamics
        # stream is untouched by message length. The protocol half (2N slots,
        # present in both conditions) is drawn from one stream and the optional
        # Q half from another, so the protocol stream also advances identically
        # in both conditions and the priming itself matches slot for slot.
        seeds = self.rng.integers(0, 2**31 - 1, size=2)
        self._obs_protocol_rng = np.random.default_rng(int(seeds[0]))
        self._obs_q_rng = np.random.default_rng(int(seeds[1]))

        self.number_of_nodes: int = len(adjacency_matrix)

        self.max_dead_nodes: int = (
            self.number_of_nodes if max_dead_nodes is None else int(max_dead_nodes)
        )
        self.states: np.ndarray = np.ones(self.number_of_nodes, dtype=np.int8)
        self.in_neighbors: list[np.ndarray] = [
            np.where(self.adjacency_matrix[:, i] == 1)[0] for i in range(self.number_of_nodes)
        ]
        # Row i = the nodes that receive i's message, i.e. the ones i's Q-slots can
        # possibly help. Used by the neighborhood reward term.
        self._out_adjacency = np.asarray(self.adjacency_matrix, dtype=np.float32)

        self.B_START, self.B_END = 0, self.number_of_nodes
        self.M_START, self.M_END = self.B_END, self.B_END + self.length_message

        self.observations: dict[int, np.ndarray] = {}

        # Current per-node loss rate, resampled every step in forward().
        self.p_loss = np.zeros(self.number_of_nodes, dtype=np.float32)

        self._initialize_observations()

    def _sample_loss_rate(self, load: int, n_neighbors: int, rng) -> float:
        """Draw a node's loss rate from the congestion kernel.

        load: number of in-neighbors that sent a real message this step.
        Returns a loss rate in [0, p_loss_max]; higher congestion (more
        talkers relative to in-degree) shifts the drawn rate upward.
        """
        if n_neighbors == 0 or load == 0:
            return 0.0

        rho = load / n_neighbors                       # congestion fraction in (0, 1]
        rho = float(np.clip(rho, 1e-6, 1.0 - 1e-6))    # keep Beta parameters valid
        kappa = self.loss_concentration
        rate = rng.beta(rho * kappa, (1.0 - rho) * kappa)
        return self.p_loss_max * rate

    def reset(self, forced_rng=None, initial_failure_rate=None):
        """
        initial_failure_rate: fraction of nodes that start dead. The training
        curriculum draws it per episode; evaluation leaves it at 0.
        """
        current_rng = forced_rng if forced_rng is not None else self.rng

        p_initial_failure = initial_failure_rate or 0.0

        self.states = np.ones(self.number_of_nodes, dtype=np.int8)
        self.p_loss[:] = 0.0

        if p_initial_failure > 0:
            failure_mask = current_rng.random(self.number_of_nodes) < p_initial_failure
            dead_nodes = np.where(failure_mask)[0]

            if dead_nodes.size > self.max_dead_nodes:
                dead_nodes = current_rng.choice(
                    dead_nodes, size=self.max_dead_nodes, replace=False
                )

            self.states[dead_nodes] = 0

            if not np.any(self.states):
                random_node = current_rng.integers(0, self.number_of_nodes)
                self.states[random_node] = 1
        
        self._initialize_observations(current_rng)
    

    def _initialize_observations(self, forced_rng=None):
        """Prime the first frame with random symbols.

        Deliberately ignores forced_rng: this noise must never be drawn from the
        stream that produces failures and packet loss, or the number of draws
        (which scales with length_message) would make the Q-slot ablation face a
        different failure sequence than the full condition. See __init__.
        """
        self.observations = {}

        n_symbols = len(self.message_candidates)
        protocol_len = min(2 * self.number_of_nodes, self.length_message)
        random_messages = np.empty(
            (self.number_of_nodes, self.length_message), dtype=np.int64
        )
        random_messages[:, :protocol_len] = self._obs_protocol_rng.integers(
            0, n_symbols, size=(self.number_of_nodes, protocol_len)
        )
        if self.length_message > protocol_len:
            random_messages[:, protocol_len:] = self._obs_q_rng.integers(
                0, n_symbols, size=(self.number_of_nodes, self.length_message - protocol_len)
            )
        random_messages[self.states == 0] = -1

        for i in range(self.number_of_nodes):
            neighbors_id = self.in_neighbors[i]

            if neighbors_id.size == 0:
                raise ValueError(f"Node {i} has no incoming neighbors, which is not allowed in this environment.")

            obs = np.empty((neighbors_id.size, self.length_message + 1), dtype=np.int8)
            obs[:, 0] = neighbors_id
            obs[:, 1:] = random_messages[neighbors_id]
            self.observations[i] = obs


    def forward(self, actions: np.ndarray, forced_rng=None):
        """One round: deliver messages (with loss), score beliefs, then crash nodes.

        Returns the per-node reward (0 for dead nodes).
        """
        if not np.any(self.states):
            raise AllNodesDeadError("All nodes are dead.")
        
        current_rng = forced_rng if forced_rng is not None else self.rng
        
        alive_mask = self.states == 1
        n_alive = np.sum(alive_mask)

        for i in np.where(alive_mask)[0]:
            neighbors_id = self.in_neighbors[i]
            if neighbors_id.size == 0:
                self.observations[i] = np.zeros((0, self.length_message + 1), dtype=np.int8)
                continue

            # Message extraction
            received = actions[neighbors_id, self.M_START:self.M_END].copy()

            # Congestion: how many in-neighbors actually spoke this step (any
            # non-silent bit), measured before loss is applied. This drives the
            # node-wise loss rate for the current step.
            load = int(np.count_nonzero(np.any(received != -1, axis=1)))
            self.p_loss[i] = self._sample_loss_rate(load, neighbors_id.size, current_rng)

            loss_mask = current_rng.random(neighbors_id.size) < self.p_loss[i]
            received[loss_mask, :] = -1

            obs = np.empty((neighbors_id.size, self.length_message + 1), dtype=np.int8)
            obs[:, 0] = neighbors_id
            obs[:, 1:] = received
            self.observations[i] = obs

        alive_beliefs = actions[alive_mask, self.B_START:self.B_END]

        errors = np.abs(self.states - alive_beliefs)
        belief_error = np.sum(errors**2, axis=1) / n_alive

        rewards = np.zeros(self.number_of_nodes, dtype=np.float32)
        if self.neighborhood_reward_weight > 0.0:
            # Mean belief error over the alive nodes that hear node i, credited back
            # to i. Vectorized as a masked adjacency average so this stays cheap in
            # the inner rollout loop.
            alive_f = alive_mask.astype(np.float32)
            err_by_node = np.zeros(self.number_of_nodes, dtype=np.float32)
            err_by_node[alive_mask] = belief_error
            listeners_error = self._out_adjacency @ (err_by_node * alive_f)
            n_listeners = self._out_adjacency @ alive_f
            neighbor_error = np.divide(
                listeners_error, n_listeners,
                out=np.zeros_like(listeners_error), where=n_listeners > 0,
            )
            shaped_error = belief_error + self.neighborhood_reward_weight * neighbor_error[alive_mask]
        else:
            shaped_error = belief_error
        rewards[alive_mask] = -shaped_error

        failure_mask = current_rng.random(self.number_of_nodes) < self.p_failure

        new_failures = np.where(failure_mask & alive_mask)[0]

        current_dead = self.number_of_nodes - n_alive
        allowed = self.max_dead_nodes - current_dead
        if new_failures.size > allowed:
            new_failures = (
                current_rng.choice(new_failures, size=allowed, replace=False)
                if allowed > 0
                else np.empty(0, dtype=int)
            )

        self.states[new_failures] = 0
        # Never let the whole graph die: revive one node picked at random.
        if not np.any(self.states):
            random_alive_node = current_rng.choice(np.where(alive_mask)[0])
            self.states[random_alive_node] = 1
        return rewards