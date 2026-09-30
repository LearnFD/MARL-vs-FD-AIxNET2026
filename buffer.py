import numpy as np

class ReplayBuffer:
    """Circular transition buffer for the recurrent network.

    Stores single-step frames plus the GRU hidden state that preceded them
    (R2D2-style stored state), so training can do a 1-step unroll from the
    rollout-time hidden state.
    """

    # Field order is the contract with sample(); push_batch takes the same names.
    FIELD_ORDER = (
        "rewards", "dones", "q_messages", "node_states", "node_ids",
        "frames", "hidden_in", "next_frames", "next_hidden",
    )

    def __init__(self, capacity, frame_dim, hidden_dim, number_of_nodes, n_q_slots, rng,
                 extra_specs=None):
        """extra_specs: optional {name: (per-item shape, dtype)} for additional
        fields. They are pushed and sampled after the base fields, in the order
        given."""
        self.capacity = capacity
        self.cursor = 0
        self.full = False
        self.rng = rng

        self.data = {
            "rewards": np.empty(capacity, dtype=np.float32),
            "dones": np.empty(capacity, dtype=np.bool_),
            "q_messages": np.empty((capacity, n_q_slots), dtype=np.int8),
            "node_states": np.empty((capacity, number_of_nodes), dtype=np.float32),
            "node_ids": np.empty(capacity, dtype=np.int64),
            "frames": np.empty((capacity, frame_dim), dtype=np.float32),
            "hidden_in": np.empty((capacity, hidden_dim), dtype=np.float32),
            "next_frames": np.empty((capacity, frame_dim), dtype=np.float32),
            "next_hidden": np.empty((capacity, hidden_dim), dtype=np.float32),
        }
        extra_specs = extra_specs or {}
        for name, (shape, dtype) in extra_specs.items():
            self.data[name] = np.empty((capacity, *shape), dtype=dtype)
        self.FIELD_ORDER = type(self).FIELD_ORDER + tuple(extra_specs)

        self.sampling_ptr = 0
        self._must_shuffle = True

    def _update_sampling_logic(self):
        bound = self.capacity if self.full else self.cursor
        self.active_indices = np.arange(bound)
        self.rng.shuffle(self.active_indices)
        self.sampling_ptr = 0
        self._must_shuffle = False

    def push_batch(self, **batch):
        batch_size = len(batch["rewards"])

        if self.cursor + batch_size <= self.capacity:
            for name in self.FIELD_ORDER:
                self.data[name][self.cursor:self.cursor + batch_size] = batch[name]

            self.cursor += batch_size
            if self.cursor == self.capacity:
                self.full = True
        else:
            first_part_size = self.capacity - self.cursor
            second_part_size = batch_size - first_part_size

            for name in self.FIELD_ORDER:
                self.data[name][self.cursor:] = batch[name][:first_part_size]
                self.data[name][:second_part_size] = batch[name][first_part_size:]

            self.cursor = second_part_size
            self.full = True
            self._must_shuffle = True

    def sample(self, batch_size):
        bound = self.capacity if self.full else self.cursor

        if bound < batch_size:
            raise ValueError("Not enough samples in buffer to sample the requested batch size.")

        if self._must_shuffle or (self.sampling_ptr + batch_size > len(self.active_indices)):
            self._update_sampling_logic()

        indices = self.active_indices[self.sampling_ptr : self.sampling_ptr + batch_size]
        self.sampling_ptr += batch_size

        return tuple(self.data[name][indices] for name in self.FIELD_ORDER)
