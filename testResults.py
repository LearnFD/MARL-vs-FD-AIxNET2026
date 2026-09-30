from abc import ABC, abstractmethod
import numpy as np

class ResultInterface(ABC):

    @abstractmethod
    def feed(self, episode: int, states: np.ndarray, beliefs: list) -> None:
        """Feed the result instance with the provided data."""
        pass




class Heatmaps(ResultInterface):
    def __init__(self, n_nodes, episodes_to_store, episode_length):
        self.n_nodes = n_nodes
        self.episodes_to_store = episodes_to_store
        self.episode_length = episode_length
        self.index = {episode: i for i, episode in enumerate(episodes_to_store)}
        self.data = np.full((len(episodes_to_store), episode_length, n_nodes, 2), np.nan, dtype=np.float32) # [average beliefs, states]
    
    def feed(self, episode: int, states: np.ndarray, beliefs: list) -> None:
        if episode not in self.index:
            return
        
        length = min(len(states), self.episode_length)
        self.data[self.index[episode], :length, :, 1] = states

        for i, belief in enumerate(beliefs):
            self.data[self.index[episode], i, :, 0] = belief.mean(axis=0)



class FP_FN(ResultInterface):
    def __init__(self, n_episodes, n_nodes, threshold):
        self.n_episodes = n_episodes
        self.n_nodes = n_nodes
        self.threshold = threshold

        self.data = np.full((n_episodes, 2), np.nan, dtype=np.float32) # [FP relative rate, FN relative rate]
    
    def feed(self, episode: int, states: np.ndarray, beliefs: list) -> None:
        alive_mask = (states == 1)
        dead_mask = (states == 0)

        fp_opportunities = np.maximum(1, np.sum(alive_mask) * self.n_nodes)
        fn_opportunities = np.maximum(1, np.sum(dead_mask) * self.n_nodes)

        fp, fn = 0, 0

        for i, belief in enumerate(beliefs):
            fp += np.sum((belief[:, alive_mask[i]] < min(self.threshold, 1 - self.threshold)))
            fn += np.sum((belief[:, dead_mask[i]] >= max(self.threshold, 1 - self.threshold)))
        
        self.data[episode, 0] = fp / fp_opportunities
        self.data[episode, 1] = fn / fn_opportunities


class AccuracyProbability(ResultInterface):
    def __init__(self, n_episodes, episode_length, threshold: float = 0.5):
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.threshold = threshold
        self.data = np.full((n_episodes, episode_length), np.nan, dtype=np.float32)

    def feed(self, episode: int, states: np.ndarray, beliefs: list) -> None:
        for i, belief in enumerate(beliefs):
            if belief.size == 0:
                continue

            truth = (states[i] == 1)                                # [n_nodes]
            correct = (belief >= self.threshold) == truth[None, :]             # [n_alive, n_nodes]
            correct = correct.astype(np.float32)

            self.data[episode, i] = np.nanmean(correct)

    def probability(self) -> float:
        return float(np.nanmean(self.data))


class DetectionTime(ResultInterface):
    def __init__(self, n_nodes, threshold: float = 0.5):
        self.n_nodes = n_nodes
        self.threshold = threshold
        self.times = []          # detection delays (in steps) for detected failures
        self.n_failures = 0
        self.n_detected = 0

    def feed(self, episode: int, states: np.ndarray, beliefs: list) -> None:
        n_steps = states.shape[0]

        for j in range(self.n_nodes):
            for t in range(1, n_steps):
                if states[t - 1, j] == 1 and states[t, j] == 0:
                    # Observers = agents alive at the death step (j itself is dead, excluded).
                    # Each observer counts once; its detection time is the FIRST step its
                    # belief about j drops below threshold.
                    observers = np.where(states[t] == 1)[0]
                    if observers.size == 0:
                        break

                    # below[k, agent] = "agent believes j is dead" at step t+k.
                    # beliefs[s] rows are the alive agents (sorted ids), so we map them
                    # back to agent ids via the per-step alive set rather than assuming a
                    # fixed observer ordering (other observers may die mid-window).
                    below = np.zeros((n_steps - t, self.n_nodes), dtype=bool)
                    for k, s in enumerate(range(t, n_steps)):
                        alive_s = np.where(states[s] == 1)[0]
                        below[k, alive_s] = beliefs[s][:, j] < self.threshold

                    obs_below = below[:, observers]              # [window, n_observers]
                    detected = obs_below.any(axis=0)             # detected at all?
                    first_delay = obs_below.argmax(axis=0)       # first True step (0 if none)

                    self.n_failures += observers.size
                    self.n_detected += int(detected.sum())
                    self.times.extend(first_delay[detected].tolist())
                    break  # only the first failure of node j per episode

    def mean_detection_time(self) -> float:
        return float(np.mean(self.times)) if self.times else float("nan")


class DetectionTimeByDistance(ResultInterface):
    """Detection delay and detection rate split by observer-to-failure hop distance.

    Same event definition as DetectionTime -- the first failure of each node per
    episode, one entry per observer alive at the death step -- except each
    observer is filed under its hop distance to the node that died. A direct
    neighbour (distance 1) stops receiving that node's keep-alive and can notice
    on its own; anything at distance >= 2 can only find out through the channel,
    so this is where a learned code has to earn its keep. Distance 0 is the dead
    node reporting on itself and is excluded.
    """

    def __init__(self, n_nodes, distances, threshold: float = 0.5):
        self.n_nodes = n_nodes
        self.threshold = threshold
        self.distances = np.asarray(distances, dtype=np.float32)

        finite = self.distances[np.isfinite(self.distances)]
        self.max_distance = int(finite.max()) if finite.size else 0

        self.times = {d: [] for d in range(1, self.max_distance + 1)}
        self.n_failures = np.zeros(self.max_distance + 1, dtype=np.int64)
        self.n_detected = np.zeros(self.max_distance + 1, dtype=np.int64)

    def feed(self, episode: int, states: np.ndarray, beliefs: list) -> None:
        n_steps = states.shape[0]

        for j in range(self.n_nodes):
            for t in range(1, n_steps):
                if states[t - 1, j] == 1 and states[t, j] == 0:
                    observers = np.where(states[t] == 1)[0]
                    if observers.size == 0:
                        break

                    below = np.zeros((n_steps - t, self.n_nodes), dtype=bool)
                    for k, s in enumerate(range(t, n_steps)):
                        alive_s = np.where(states[s] == 1)[0]
                        below[k, alive_s] = beliefs[s][:, j] < self.threshold

                    obs_below = below[:, observers]
                    detected = obs_below.any(axis=0)
                    first_delay = obs_below.argmax(axis=0)

                    for k, observer in enumerate(observers):
                        distance = self.distances[observer, j]
                        if not np.isfinite(distance) or distance < 1:
                            continue
                        distance = int(distance)
                        self.n_failures[distance] += 1
                        if detected[k]:
                            self.n_detected[distance] += 1
                            self.times[distance].append(int(first_delay[k]))
                    break  # only the first failure of node j per episode

    def summary(self) -> dict:
        out = {}
        for d in range(1, self.max_distance + 1):
            times = np.asarray(self.times[d], dtype=np.float64)
            n_pairs = int(self.n_failures[d])
            out[str(d)] = {
                "n_observer_failure_pairs": n_pairs,
                "n_detected": int(self.n_detected[d]),
                "detection_rate": (float(self.n_detected[d]) / n_pairs) if n_pairs else float("nan"),
                "mean_detection_time_steps": float(times.mean()) if times.size else float("nan"),
                "std_detection_time_steps": float(times.std(ddof=1)) if times.size > 1 else float("nan"),
                "median_detection_time_steps": float(np.median(times)) if times.size else float("nan"),
                "p90_detection_time_steps": float(np.percentile(times, 90)) if times.size else float("nan"),
            }
        return out


class BeliefsByDistance(ResultInterface):
    """Mean reported belief and correctness, split by observer-to-target hop distance.

    Accumulated over every (step, observer, target) triple, with the target's
    true state as the split: 'alive' buckets should sit near 1 and 'dead' near 0,
    so the *gap* between them at a given distance is how much an observer that
    far out actually knows. Self-reports (distance 0) are excluded, since a node
    is trivially right about itself.
    """

    def __init__(self, n_nodes, distances, threshold: float = 0.5):
        self.n_nodes = n_nodes
        self.threshold = threshold
        self.distances = np.asarray(distances, dtype=np.float32)

        finite = self.distances[np.isfinite(self.distances)]
        self.max_distance = int(finite.max()) if finite.size else 0
        size = self.max_distance + 1

        self.sum_alive = np.zeros(size, dtype=np.float64)
        self.sum_dead = np.zeros(size, dtype=np.float64)
        self.count_alive = np.zeros(size, dtype=np.int64)
        self.count_dead = np.zeros(size, dtype=np.int64)
        self.correct_alive = np.zeros(size, dtype=np.float64)
        self.correct_dead = np.zeros(size, dtype=np.float64)

    def feed(self, episode: int, states: np.ndarray, beliefs: list) -> None:
        for s, belief in enumerate(beliefs):
            if belief.size == 0:
                continue

            observers = np.where(states[s] == 1)[0]
            # beliefs[s] rows are exactly the agents alive at step s, in sorted
            # id order (see iql.test); bail out rather than mis-attribute if that
            # ever stops holding.
            if observers.size != belief.shape[0]:
                continue

            distance = self.distances[observers]                    # [n_obs, n_nodes]
            valid = np.isfinite(distance) & (distance >= 1)
            bucket = np.where(valid, distance, 0).astype(np.int64)

            target_alive = (states[s] == 1)[None, :] & valid
            target_dead = (states[s] == 0)[None, :] & valid
            correct = ((belief >= self.threshold) == (states[s] == 1)[None, :])

            np.add.at(self.sum_alive, bucket[target_alive], belief[target_alive])
            np.add.at(self.count_alive, bucket[target_alive], 1)
            np.add.at(self.correct_alive, bucket[target_alive], correct[target_alive])

            np.add.at(self.sum_dead, bucket[target_dead], belief[target_dead])
            np.add.at(self.count_dead, bucket[target_dead], 1)
            np.add.at(self.correct_dead, bucket[target_dead], correct[target_dead])

    def summary(self) -> dict:
        out = {}
        for d in range(1, self.max_distance + 1):
            n_alive = int(self.count_alive[d])
            n_dead = int(self.count_dead[d])
            mean_alive = (self.sum_alive[d] / n_alive) if n_alive else float("nan")
            mean_dead = (self.sum_dead[d] / n_dead) if n_dead else float("nan")
            n_total = n_alive + n_dead
            out[str(d)] = {
                "n_alive_target_pairs": n_alive,
                "n_dead_target_pairs": n_dead,
                "mean_belief_alive_target": float(mean_alive),
                "mean_belief_dead_target": float(mean_dead),
                "belief_gap": float(mean_alive - mean_dead) if (n_alive and n_dead) else float("nan"),
                "accuracy_alive_target": float(self.correct_alive[d] / n_alive) if n_alive else float("nan"),
                "accuracy_dead_target": float(self.correct_dead[d] / n_dead) if n_dead else float("nan"),
                "accuracy": (float(self.correct_alive[d] + self.correct_dead[d]) / n_total) if n_total else float("nan"),
            }
        return out




class testResults:

    REGISTRY = {
        "heatmaps": Heatmaps,
        "accuracy_probability": AccuracyProbability,
        "detection_time": DetectionTime,
        "detection_time_by_distance": DetectionTimeByDistance,
        "beliefs_by_distance": BeliefsByDistance,
        "fp_fn": FP_FN,
    }

    def __init__(self, results_config, episode_length, n_nodes):
        self.episode_length = episode_length
        self.n_nodes = n_nodes

        self.store_data = False

        for (key, config) in results_config.items():
            if config:
                self.store_data = True
                setattr(self, key, self._create(key, config))
        
        if self.store_data:
            self.count = 0
            self._states = np.empty((self.episode_length, self.n_nodes), dtype=np.int8)
            self._beliefs = []


    def _create(self, key, config):
        cls = self.REGISTRY.get(key)

        if cls is None:
            raise ValueError(f"Unknown result type: {key}")
        
        try:
            return cls(**config)
        except TypeError as e:
            raise ValueError(f"Invalid config for {key}: {e}")


    def store(self, states: np.ndarray, beliefs: np.ndarray):
        if self.store_data:

            if self.count >= self.episode_length:
                raise ValueError("Episode length exceeded during storage.")

            self._states[self.count] = states
            self._beliefs.append(beliefs)
            
            self.count += 1


    def update(self, episode: int):
        if self.store_data:
            active_states = self._states[:self.count]
            active_beliefs = self._beliefs[:self.count]

            for instance_name in self.REGISTRY.keys():
                result_instance = getattr(self, instance_name, None)

                if result_instance is not None:
                    result_instance.feed(
                        episode=episode,
                        states=active_states,
                        beliefs=active_beliefs
                    )
            
            self.count = 0

            self._states[:, :] = -1
            self._beliefs = []