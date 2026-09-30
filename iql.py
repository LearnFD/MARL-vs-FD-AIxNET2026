import time
import numpy as np
import torch
import copy
import torch.nn.functional as F
import torch.optim as optim

from env import AllNodesDeadError, Environment
from network import Network
from buffer import ReplayBuffer
from testResults import testResults
from trainResults import trainResults

class IqlAlgorithms:
    def __init__(self, env: Environment, network: Network, rng: np.random.Generator):
        self.env = env
        self.network = network

        # Set the RNG for the algorithm (training and testing)
        self.rng = rng

        # Useful parameters for the algorithms
        self.messages_candidates = np.array(network.message_candidates)
        self.n_nodes = env.number_of_nodes
        self.L = env.length_message

        self.n_age_slots: int = network.n_age_slots
        self.n_q_slots: int = network.n_q_slots
        self.age_cap: float = network.age_cap
        self.hidden_dim: int = network.hidden_dim

        # Per-neighbor message rows only — the recipient's identity is supplied
        # separately (one-hot conditioning) rather than baked into the frame.
        self.preprocessed_obs = np.full((self.n_nodes - 1, self.L), -2, dtype=np.float32)

        self.dead_action = np.concatenate([np.full(self.n_nodes, np.nan), np.full(self.L, -1)])

        # Protocol state, reset every episode:
        #   ages[i, j]  = agent i's heartbeat age about node j (min-plus relayed)
        #   hiddens[i]  = agent i's GRU hidden state
        self.ages = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float32)
        self.hiddens = np.zeros((self.n_nodes, self.hidden_dim), dtype=np.float32)

    def get_rng(self, seed: int):
        return np.random.default_rng(seed=seed)

    # ------------------------------------------------------------------
    # Age-of-information protocol (hand-coded, not learned): each agent keeps
    # a per-node heartbeat age and relays it in its messages. Receiving a real
    # message from s is direct evidence s is alive (age 0); relayed evidence
    # about j costs one hop (reported age + 1). Ages otherwise grow by 1 per
    # step and saturate at age_cap — a saturating age is exactly the signal a
    # gossip failure detector thresholds on, here handed to the network as input.
    # ------------------------------------------------------------------
    def _reset_protocol_state(self):
        self.ages[:, :] = 0.0
        self.hiddens[:, :] = 0.0
        np.fill_diagonal(self.ages, 0.0)

    def _update_ages(self):
        if self.n_age_slots == 0:
            return  # learned channel: no age block on the wire, nothing to relay
        na = self.n_age_slots
        alive_nodes = np.where(self.env.states == 1)[0]
        for i in alive_nodes:
            ages_i = np.minimum(self.ages[i] + 1.0, self.age_cap)
            for row in self.env.observations[i]:
                sender_id = int(row[0])
                message = row[1:]
                if np.all(message == -1):
                    continue  # silent: dead sender or lost packet, no evidence
                candidate = message[:na].astype(np.float32) + 1.0
                candidate[sender_id] = 0.0  # hearing s at all is a direct heartbeat
                ages_i = np.minimum(ages_i, candidate)
            ages_i[i] = 0.0
            self.ages[i] = np.minimum(ages_i, self.age_cap)

    def preprocess_observation(self, node_id: int, init: bool = False) -> np.ndarray:
        na = self.n_age_slots
        result = self.preprocessed_obs.copy()
        obs = self.env.observations[node_id]
        if obs.size > 0:
            for i in range(obs.shape[0]):
                sender_id = int(obs[i, 0])
                row = sender_id if sender_id < node_id else sender_id - 1
                result[row, :] = obs[i, 1:]
                if init and not np.all(obs[i, 1:] == -1):
                    # Boot prior: everything heard about is fresh. Only applies to
                    # the age block; learned slots have no such convention.
                    if na > 0:
                        result[row, :na] = 0
        return result

    def build_frames(self, node_ids, init: bool = False) -> np.ndarray:
        return np.stack([
            np.concatenate([
                self.preprocess_observation(node_id, init).ravel(),
                self.ages[node_id],
            ])
            for node_id in node_ids
        ]).astype(np.float32)

    def choose_random_q_actions(self, n_actions):
        return self.rng.integers(
            len(self.messages_candidates), size=(n_actions, self.n_q_slots)
        ).astype(np.int64)

    def initialize_actions(self):
        actions = np.full((self.n_nodes, self.n_nodes + self.L), np.nan, dtype=np.float32)
        actions[:, self.n_nodes:] = -1
        return actions

    def write_actions(self, actions, node_ids, belief_probs, q_ids):
        n = self.n_nodes
        na = self.n_age_slots
        # Own belief estimate for the reward/accuracy calc (env.py's B_START:B_END)
        # is always written — it is never broadcast, only scored.
        actions[node_ids, :n] = belief_probs
        offset = n
        if na > 0:
            actions[node_ids, offset:offset + na] = self.ages[node_ids]
            offset += na
        actions[node_ids, offset:] = self.messages_candidates[q_ids]

    def _listener_context(self, alive_nodes, next_frames, out_neighbors, max_listeners, frame_dim):
        """Round-(t+1) inputs of each emitter's listeners, for the counterfactual.

        For emitter i (alive at t) and each neighbor j still alive at t+1: j's
        next frame and GRU state, where i's learned block starts in that frame,
        and whether i's message actually reached j this round (a lost or dead
        row carries no symbols, so changing them would change nothing).
        """
        n = alive_nodes.size
        L, na = self.L, self.n_age_slots
        pos_of = {int(node): k for k, node in enumerate(alive_nodes)}
        ctx = {
            "lis_frames": np.zeros((n, max_listeners, frame_dim), dtype=np.float32),
            "lis_hidden": np.zeros((n, max_listeners, self.hidden_dim), dtype=np.float32),
            "lis_ids": np.zeros((n, max_listeners), dtype=np.int64),
            "lis_pos": np.zeros((n, max_listeners), dtype=np.int64),
            "lis_mask": np.zeros((n, max_listeners), dtype=np.bool_),
            "next_truth": np.tile(self.env.states.astype(np.float32), (n, 1)),
        }
        for p, i in enumerate(alive_nodes):
            for d, j in enumerate(out_neighbors[int(i)]):
                if self.env.states[j] != 1 or int(j) not in pos_of:
                    continue
                q = pos_of[int(j)]
                row = int(i) if i < j else int(i) - 1
                block = next_frames[q, row * L:(row + 1) * L]
                ctx["lis_frames"][p, d] = next_frames[q]
                ctx["lis_hidden"][p, d] = self.hiddens[j]
                ctx["lis_ids"][p, d] = j
                ctx["lis_pos"][p, d] = row * L + na
                ctx["lis_mask"][p, d] = not np.all(block == -1)
        return ctx

    def _counterfactual_advantage(self, extras):
        """Advantage of every (slot, symbol) for the emitters of a replay batch.

        Returns advantage [B, n_q, C] -- minus the listeners' mean squared belief
        error at t+1 with that one symbol substituted, centred over the alphabet
        and scaled to unit spread over the batch -- and a [B] mask of emitters
        with at least one listener that received the message.
        """
        lis_frames, lis_hidden, lis_ids, lis_pos, lis_mask, next_truth = (
            torch.as_tensor(a, device=self.network.device) for a in extras
        )
        B, D, F_ = lis_frames.shape
        K, C = self.n_q_slots, len(self.messages_candidates)
        with torch.no_grad():
            x = lis_frames[:, :, None, None, :].expand(B, D, K, C, F_).clone()
            col = (lis_pos[:, :, None, None] + torch.arange(K, device=x.device)[None, None, :, None])
            col = col.expand(B, D, K, C)
            sym = torch.as_tensor(self.messages_candidates, dtype=x.dtype, device=x.device)
            sym = sym[None, None, None, :].expand(B, D, K, C)
            x.scatter_(-1, col.unsqueeze(-1), sym.unsqueeze(-1))
            h = lis_hidden[:, :, None, None, :].expand(B, D, K, C, lis_hidden.size(-1))
            ids = lis_ids[:, :, None, None].expand(B, D, K, C)
            logits, _, _ = self.network(
                x.reshape(-1, F_), h.reshape(-1, h.size(-1)), ids.reshape(-1)
            )
            probs = torch.sigmoid(logits).view(B, D, K, C, -1)
            err = ((probs - next_truth[:, None, None, None, :]) ** 2).sum(dim=-1)
            mask = lis_mask[:, :, None, None].float()
            err = (err * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)   # [B, K, C]
            adv = -(err - err.mean(dim=-1, keepdim=True))
            has_listener = lis_mask.any(dim=1)
            if has_listener.any():
                adv = adv / adv[has_listener].std().clamp(min=1e-8)
        return adv, has_listener

    def train(self, train_config: dict):
        """Train the belief head (supervised BCE) and the message head (IQL).

        train_config keys: learning_rate, exploration, update_frequency,
        target_update_frequency, number_of_episodes, episode_length, batch_size,
        max_buffer_size, and optionally warmup_steps (500), dead_class_weight
        (4.0), curriculum_max_initial_failure (0.25), gamma (0.75),
        msg_lr_scale (0.1), checkpoint_fn / checkpoint_every.

        Returns a trainResults holding the per-episode training curves.
        """
        belief_params = (
            list(self.network.q_embedding.parameters())
            + list(self.network.belief_encoder.parameters())
            + list(self.network.q_encoder.parameters())
            + list(self.network.core.parameters())
            + list(self.network.belief_head.parameters())
        )
        belief_optimizer = optim.Adam(belief_params, lr=train_config["learning_rate"])
        # Q-slot ablation (n_q_slots == 0): there is no learned message channel, so
        # the message head / adapter are never used and no message optimizer exists.
        if self.n_q_slots > 0:
            message_params = (
                list(self.network.message_head.parameters())
                + list(self.network.msg_adapter.parameters())
            )
            # The message head gets 0.1x the belief lr by default. Over a short run
            # that is very few effective updates (one per update_frequency steps) at
            # a tiny step size, which on its own is enough to leave the channel
            # untrained; exposed so it can be raised.
            message_optimizer = optim.Adam(
                message_params,
                lr=train_config["learning_rate"] * train_config.get("msg_lr_scale", 0.1)
            )
        else:
            message_params = []
            message_optimizer = None
        warmup_steps = train_config.get("warmup_steps", 500)
        # Dead targets are rare (class imbalance grows with N); upweight them so the
        # BCE isn't minimized by hugging the alive prior.
        dead_class_weight = train_config.get("dead_class_weight", 4.0)
        # Curriculum: episodes start with a random fraction of nodes already dead so
        # the network sees dead-at-various-ages patterns, not only rare fresh deaths.
        curriculum_max_initial_failure = train_config.get("curriculum_max_initial_failure", 0.25)
        target_network = copy.deepcopy(self.network)
        target_network.eval()
        for param in target_network.parameters():
            param.requires_grad = False
        # A message only pays off through a loop that takes at least two hops
        # (i speaks -> j's belief improves -> j rebroadcasts -> i benefits), so a
        # short horizon puts the entire payoff of communicating out of sight.
        # gamma=0.75 sees ~4 steps; episodes are 30 long.
        gamma = train_config.get("gamma", 0.75)
        adjacency = np.asarray(self.env.adjacency_matrix) == 1
        # How the message head is trained.
        #   "td"                one-step Q-learning on r_i(t), the agent's own belief
        #                       error at round t (the submitted paper). That reward is
        #                       fixed before m_i(t) is chosen, so it cannot depend on
        #                       the message, and the code stays as initialized.
        #   "counterfactual_td" per-slot Q-learning on a difference reward (the
        #                       second version of the paper). For the symbol actually
        #                       sent, the reward is the listeners' round-(t+1) belief
        #                       error averaged over the alphabet minus their error with
        #                       that symbol, computed by replaying their recorded inputs
        #                       through the current belief head. Loss, crashes and the
        #                       other senders are identical in both terms and cancel.
        msg_objective = train_config.get("msg_objective", "td")
        if msg_objective not in ("td", "counterfactual_td"):
            raise ValueError(f"unknown msg_objective {msg_objective!r}")
        counterfactual = msg_objective == "counterfactual_td" and self.n_q_slots > 0
        out_neighbors = [np.flatnonzero(adjacency[i]) for i in range(self.n_nodes)]
        max_listeners = max(len(nb) for nb in out_neighbors)
        frame_dim = self.network.base_input_dim

        replay_buffer = ReplayBuffer(
            capacity=train_config["max_buffer_size"],
            frame_dim=self.network.base_input_dim,
            hidden_dim=self.hidden_dim,
            number_of_nodes=self.n_nodes,
            n_q_slots=self.n_q_slots,
            rng=self.rng,
            extra_specs=({
                "lis_frames": ((max_listeners, frame_dim), np.float32),
                "lis_hidden": ((max_listeners, self.hidden_dim), np.float32),
                "lis_ids": ((max_listeners,), np.int64),
                "lis_pos": ((max_listeners,), np.int64),
                "lis_mask": ((max_listeners,), np.bool_),
                "next_truth": ((self.n_nodes,), np.float32),
            } if counterfactual else None),
        )
        step_count = 0
        last_belief_loss = 0
        last_msg_loss = 0
        start_time = time.time()

        # --- Channel diagnostics -------------------------------------------------
        # Two numbers decide whether the learned channel is alive at all:
        #   q_spread  = mean_slot(max_a Q(s,a) - mean_a Q(s,a)). If it collapses to
        #               ~0 the Q-function is flat across candidates, the argmax that
        #               picks the emitted symbol is arbitrary, and the slots carry
        #               noise regardless of how nicely msg_loss converges.
        #   entropy   = entropy of the *greedy* symbols actually emitted, in bits
        #               (max = log2(n_candidates)). Measured before the exploration
        #               mask is applied, otherwise epsilon alone would inflate it.
        # Both are sampled over a window and flushed at every progress print.
        diag_symbol_counts = np.zeros((max(self.n_q_slots, 1), len(self.messages_candidates)), dtype=np.float64)
        diag_spreads: list[float] = []
        diag_history: dict[str, list] = {
            "episode": [], "q_spread": [], "symbol_entropy_bits": [],
        }
        last_q_spread = float("nan")
        last_entropy = float("nan")

        # Results storage initialization
        results = trainResults(train_config["number_of_episodes"])

        # Epsilon decay
        epsilon_min = 0.05
        epsilon_decay = (train_config["exploration"] - epsilon_min) / max(1, train_config["number_of_episodes"]-1)

        # Optional crash insurance, wired up by run_experiment.py.
        checkpoint_fn = train_config.get("checkpoint_fn")
        checkpoint_every = max(1, int(train_config.get("checkpoint_every", 5000)))

        # Resume from a checkpoint_partial.pt (run_experiment.py --resume). The
        # partial is written at the start of an episode, before it is played, so
        # training restarts at that episode. Weights are loaded by the caller;
        # optimizer states, curves, diagnostics and the round counter are
        # restored here. The replay buffer is not saved, so it refills, and the
        # caller reseeds the random streams: a resumed run is a valid run, not a
        # bit-identical continuation.
        start_episode = 0
        resume = train_config.get("resume")
        if resume is not None:
            start_episode = int(resume["episode"]) - 1
            belief_optimizer.load_state_dict(resume["optimizer_states"]["belief"])
            if message_optimizer is not None and resume["optimizer_states"].get("message") is not None:
                message_optimizer.load_state_dict(resume["optimizer_states"]["message"])
            for name, curve in resume.get("train_curves", {}).items():
                if curve is not None and hasattr(results, name):
                    getattr(results, name).data[:start_episode] = curve[:start_episode]
            for key in diag_history:
                diag_history[key] = list(resume.get("diagnostics", {}).get(key, []))
            # Episodes always run their full length in training (a node is revived
            # rather than letting the whole graph die), so the counter is exact.
            step_count = start_episode * train_config["episode_length"]
            target_network.load_state_dict(self.network.state_dict())
            print(f"[resume] restarting at episode {start_episode} "
                  f"(step {step_count})", flush=True)

        for e in range(start_episode, train_config["number_of_episodes"]):
            p_initial = (
                self.rng.random() * curriculum_max_initial_failure
                if curriculum_max_initial_failure > 0
                else None
            )
            self.env.reset(initial_failure_rate=p_initial)
            self._reset_protocol_state()

            if (e + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                if self.n_q_slots > 0:
                    if diag_spreads:
                        last_q_spread = float(np.mean(diag_spreads))
                        diag_spreads.clear()
                    totals = diag_symbol_counts.sum(axis=1, keepdims=True)
                    if totals.sum() > 0:
                        probs = np.divide(
                            diag_symbol_counts, totals,
                            out=np.zeros_like(diag_symbol_counts), where=totals > 0,
                        )
                        with np.errstate(divide="ignore", invalid="ignore"):
                            log_p = np.where(probs > 0, np.log2(probs), 0.0)
                        per_slot_entropy = -(probs * log_p).sum(axis=1)
                        last_entropy = float(per_slot_entropy[totals[:, 0] > 0].mean())
                        diag_symbol_counts[:, :] = 0.0
                    diag_history["episode"].append(e + 1)
                    diag_history["q_spread"].append(last_q_spread)
                    diag_history["symbol_entropy_bits"].append(last_entropy)
                print(
                    f"[Train] episode {e+1}/{train_config['number_of_episodes']} "
                    f"| step {step_count} | elapsed {elapsed:.1f}s | {time.strftime('%H:%M:%S')} "
                    f"| belief_loss {last_belief_loss:.4f} | msg_loss {last_msg_loss:.4f}"
                    f" | q_spread {last_q_spread:.4f} | sym_entropy {last_entropy:.3f}b",
                    # Without this, stdout redirected to a log file is block-buffered
                    # and ~47 ticks (one 8 KB block) accumulate before anything is
                    # readable -- a long run looks hung from the outside for hours.
                    flush=True,
                )

            # Periodic checkpoint. Weights used to be written once, after the last
            # episode, so a node failure or a pre-emption at episode 49,000 of
            # 50,000 threw away the entire run. The caller supplies the writer
            # because it owns the run metadata; everything episode-local is passed
            # out here.
            if checkpoint_fn is not None and (e + 1) % checkpoint_every == 0:
                checkpoint_fn(
                    episode=e + 1,
                    optimizer_states={
                        "belief": belief_optimizer.state_dict(),
                        "message": (message_optimizer.state_dict()
                                    if message_optimizer is not None else None),
                    },
                    diagnostics=diag_history,
                    train_results=results,
                )

            current_epsilon = max(epsilon_min, train_config["exploration"] - (e * epsilon_decay))

            actions = self.initialize_actions()

            for step_index in range(train_config["episode_length"]):
                step_count += 1

                alive_nodes = np.where(self.env.states == 1)[0]

                # Snapshots for the replay buffer and raw data analysis
                frames_snapshot = self.build_frames(alive_nodes, init=(step_index == 0))
                hidden_snapshot = self.hiddens[alive_nodes].copy()
                true_states_snapshot = np.tile(self.env.states.astype(np.float32), (alive_nodes.size, 1))

                # Network inference for every alive node. Beliefs (and the relayed
                # ages) always come from the protocol — only the free Q-slots are
                # explored, so exploration doesn't corrupt the belief channel that
                # other agents learn to read.
                self.network.eval()
                belief_probs, q_ids, h_out = self.network.get_actions_batch(
                    frames_snapshot, hidden_snapshot, alive_nodes
                )
                self.network.train()
                self.hiddens[alive_nodes] = h_out

                # Greedy symbols, sampled before exploration overwrites them.
                if self.n_q_slots > 0 and q_ids.size > 0:
                    for slot in range(self.n_q_slots):
                        np.add.at(diag_symbol_counts[slot], q_ids[:, slot], 1.0)

                exploration_mask = self.rng.random(alive_nodes.size) < current_epsilon
                if exploration_mask.any():
                    q_ids[exploration_mask] = self.choose_random_q_actions(int(exploration_mask.sum()))

                self.write_actions(actions, alive_nodes, belief_probs, q_ids)

                # Update actions for dead nodes
                dead_nodes = np.where(self.env.states == 0)[0]
                actions[dead_nodes, :] = self.dead_action

                # Environment step
                try:
                    reward = self.env.forward(actions)
                except AllNodesDeadError:
                    break

                # Relay heartbeat ages from the new observations, then build the
                # post-step frames for the same acting nodes (nodes that just died
                # keep stale frames — their transitions are masked by done).
                self._update_ages()
                next_frames = self.build_frames(alive_nodes)
                next_hidden = self.hiddens[alive_nodes].copy()

                dones = ((self.env.states[alive_nodes] == 0) | (step_index == train_config["episode_length"] - 1)).astype(np.float32)

                # Only the message Q-learning consumes this reward (the belief head is
                # trained by BCE).
                transition = dict(
                    rewards=reward[alive_nodes].astype(np.float32),
                    dones=dones,
                    q_messages=q_ids.astype(np.int8),
                    node_states=true_states_snapshot,
                    node_ids=alive_nodes.astype(np.int64),
                    frames=frames_snapshot,
                    hidden_in=hidden_snapshot,
                    next_frames=next_frames,
                    next_hidden=next_hidden,
                )

                if counterfactual:
                    transition.update(self._listener_context(
                        alive_nodes, next_frames, out_neighbors, max_listeners, frame_dim
                    ))
                replay_buffer.push_batch(**transition)

                # Training step
                if (step_count % train_config["update_frequency"] == 0) and (replay_buffer.full or replay_buffer.cursor >= train_config["batch_size"]):
                    batch = replay_buffer.sample(train_config["batch_size"])
                    (batch_rewards, batch_dones, batch_q_messages, batch_node_states,
                     batch_node_ids, batch_frames, batch_hidden_in,
                     batch_next_frames, batch_next_hidden) = batch[:9]

                    tensor_frames = torch.as_tensor(batch_frames, dtype=torch.float32, device=self.network.device)
                    tensor_hidden_in = torch.as_tensor(batch_hidden_in, dtype=torch.float32, device=self.network.device)
                    tensor_next_frames = torch.as_tensor(batch_next_frames, dtype=torch.float32, device=self.network.device)
                    tensor_next_hidden = torch.as_tensor(batch_next_hidden, dtype=torch.float32, device=self.network.device)
                    tensor_q_messages = torch.as_tensor(batch_q_messages, dtype=torch.long, device=self.network.device)
                    tensor_rewards = torch.as_tensor(batch_rewards, dtype=torch.float32, device=self.network.device)
                    tensor_dones = torch.as_tensor(batch_dones, dtype=torch.float32, device=self.network.device)
                    tensor_node_states = torch.as_tensor(batch_node_states, dtype=torch.float32, device=self.network.device)
                    tensor_node_ids = torch.as_tensor(batch_node_ids, dtype=torch.long, device=self.network.device)

                    belief_logits, q_message, _ = self.network(
                        tensor_frames, tensor_hidden_in, tensor_node_ids, detach_beliefs=True
                    )

                    # Class-weighted BCE, with the agent's own slot masked out (it is
                    # alive by construction and broadcasts 1.0 anyway).
                    bce = F.binary_cross_entropy_with_logits(
                        belief_logits, tensor_node_states, reduction="none"
                    )
                    weights = 1.0 + (dead_class_weight - 1.0) * (1.0 - tensor_node_states)
                    weights[torch.arange(weights.size(0), device=weights.device), tensor_node_ids] = 0.0
                    belief_loss = (bce * weights).sum() / weights.sum().clamp(min=1.0)
                    last_belief_loss = belief_loss.item()

                    if counterfactual and step_count > warmup_steps:
                        advantage, has_listener = self._counterfactual_advantage(batch[9:])
                        sent = tensor_q_messages.unsqueeze(-1)
                        # Per-slot difference reward of the symbol actually sent.
                        slot_reward = torch.gather(advantage, 2, sent).squeeze(-1)          # [B, K]
                        chosen = torch.gather(q_message, 2, sent).squeeze(-1)               # [B, K]
                        with torch.no_grad():
                            _, next_q, _ = target_network(
                                tensor_next_frames, tensor_next_hidden, tensor_node_ids,
                                detach_beliefs=True,
                            )
                            y = slot_reward + gamma * (1.0 - tensor_dones)[:, None] * next_q.max(dim=2).values
                        weight = has_listener[:, None].float()
                        msg_loss = (((chosen - y) ** 2) * weight).sum() / (weight.sum() * self.n_q_slots).clamp(min=1.0)
                        last_msg_loss = msg_loss.item()
                        with torch.no_grad():
                            msg_q_values = chosen.sum(dim=1)
                            y = y.sum(dim=1)
                            spread = (q_message.max(dim=2).values - q_message.mean(dim=2)).mean()
                            diag_spreads.append(spread.item())
                    elif self.n_q_slots > 0 and step_count > warmup_steps:
                        chosen_q = torch.gather(q_message, dim=2, index=tensor_q_messages.unsqueeze(-1)).squeeze(-1)
                        with torch.no_grad():
                            next_q_max = target_network.max_q(
                                tensor_next_frames, tensor_next_hidden, tensor_node_ids
                            )
                        # One scalar per message: the slot values are summed, so all
                        # slots share a single TD error. That is the weak per-slot
                        # credit the paper reports, not an oversight.
                        msg_q_values = chosen_q.sum(dim=1)
                        with torch.no_grad():
                            y = tensor_rewards + gamma * (1.0 - tensor_dones) * next_q_max
                        msg_loss = F.mse_loss(msg_q_values, y)
                        last_msg_loss = msg_loss.item()
                        with torch.no_grad():
                            spread = (q_message.max(dim=2).values - q_message.mean(dim=2)).mean()
                            diag_spreads.append(spread.item())

                    # Two separate backward passes: the message head reads a
                    # detached recurrent state, so msg_loss never reaches the
                    # parameters belief_optimizer owns.
                    belief_optimizer.zero_grad()
                    belief_loss.backward()
                    torch.nn.utils.clip_grad_norm_(belief_params, max_norm=1.0)
                    belief_optimizer.step()

                    if self.n_q_slots > 0 and step_count > warmup_steps:
                        message_optimizer.zero_grad()
                        msg_loss.backward()
                        torch.nn.utils.clip_grad_norm_(message_params, max_norm=1.0)
                        message_optimizer.step()

                    results.store_training(belief_loss=last_belief_loss)
                    if self.n_q_slots > 0 and step_count > warmup_steps:
                        results.store_training(
                            msg_loss=last_msg_loss,
                            chosen_q=msg_q_values.mean().item(),
                            target_q=y.mean().item()
                        )

                # Update the target network periodically
                if step_count % train_config["target_update_frequency"] == 0:
                    target_network.load_state_dict(self.network.state_dict())

            results.update(e)

        # Attached rather than returned so the existing callers of train(), which
        # all expect a bare trainResults, keep working unchanged.
        self.channel_diagnostics = diag_history

        return results


    def test(self, test_config: dict, results_config: dict, seed: int,
             max_dead_nodes: int = None):
        """Greedy rollouts on a held-out failure sequence drawn from `seed`.

        test_config: {"number_of_episodes": int, "episode_length": int}
        results_config: {metric name: constructor kwargs}, see testResults.REGISTRY.
        Returns the testResults object.
        """
        if max_dead_nodes is not None:
            self.env.max_dead_nodes = max_dead_nodes

        self.network.eval()

        test_rng = self.get_rng(seed=seed)

        results = testResults(results_config, test_config["episode_length"], self.n_nodes)

        step_count = 0
        start_time = time.time()

        for e in range(test_config["number_of_episodes"]):
            self.env.reset(forced_rng=test_rng)
            self._reset_protocol_state()

            if (e + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Test] episode {e+1}/{test_config['number_of_episodes']} "
                    f"| step {step_count} | elapsed {elapsed:.1f}s | {time.strftime('%H:%M:%S')}"
                )

            actions = self.initialize_actions()

            for step_index in range(test_config["episode_length"]):
                step_count += 1

                alive_nodes = np.where(self.env.states == 1)[0]

                if alive_nodes.size > 0:
                    frames = self.build_frames(alive_nodes, init=(step_index == 0))
                    belief_probs, q_ids, h_out = self.network.get_actions_batch(
                        frames, self.hiddens[alive_nodes], alive_nodes
                    )
                    self.hiddens[alive_nodes] = h_out
                    self.write_actions(actions, alive_nodes, belief_probs, q_ids)

                states_snapshot = self.env.states.copy()

                try:
                    self.env.forward(actions, forced_rng=test_rng)
                except AllNodesDeadError:
                    break

                results.store(
                    states=states_snapshot,
                    beliefs=actions[alive_nodes, :self.n_nodes].copy()
                )

                # Update actions for dead nodes
                dead_nodes = np.where(self.env.states == 0)[0]
                actions[dead_nodes, :] = self.dead_action

                self._update_ages()

            results.update(e)

        # End-of-test summary computed from the result object.
        acc_prob = getattr(results, "accuracy_probability", None)
        if acc_prob is not None:
            print(f"[Test] Probability of accuracy (node right about another node): {acc_prob.probability():.4f}")

        det_time = getattr(results, "detection_time", None)
        if det_time is not None:
            print(
                f"[Test] Mean detection time: {det_time.mean_detection_time():.2f} steps "
                f"({det_time.n_detected}/{det_time.n_failures} failures detected)"
            )

        return results
