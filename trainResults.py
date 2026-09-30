import numpy as np


class PerEpisodeMean:
    """Mean of one training signal over the optimizer updates of each episode.

    NaN for episodes with no update of that kind (before the replay buffer fills,
    or before the message warmup ends).
    """
    def __init__(self, n_episodes):
        self.n_episodes = n_episodes
        self.data = np.full(n_episodes, np.nan, dtype=np.float32)

    def feed(self, episode: int, values: list) -> None:
        if len(values) > 0:
            self.data[episode] = float(np.mean(values))


class trainResults:
    """The four training curves of the paper's training-dynamics figure:
    belief loss, message TD loss, and the chosen vs target message Q."""

    SIGNALS = ("belief_loss", "msg_loss", "mean_chosen_q", "mean_target_q")

    def __init__(self, n_episodes):
        for name in self.SIGNALS:
            setattr(self, name, PerEpisodeMean(n_episodes))
        self._pending = {name: [] for name in self.SIGNALS}

    def store_training(self, belief_loss: float = None, msg_loss: float = None,
                       chosen_q: float = None, target_q: float = None):
        for name, value in (("belief_loss", belief_loss), ("msg_loss", msg_loss),
                            ("mean_chosen_q", chosen_q), ("mean_target_q", target_q)):
            if value is not None:
                self._pending[name].append(value)

    def update(self, episode: int):
        for name in self.SIGNALS:
            getattr(self, name).feed(episode, self._pending[name])
            self._pending[name] = []
