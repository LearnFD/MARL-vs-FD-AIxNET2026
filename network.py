import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: x + MLP(LayerNorm(x)). Preserves width so
    blocks can be stacked to add depth without hurting gradient flow."""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class Network(nn.Module):
    """Recurrent, age-aware belief/message network.

    Message layout (length_message = n_age_slots + n_q_slots, each block only
    present if its slot count is > 0):
        [0, n_age_slots)        age slots, 0..age_cap: steps since the sender
                                 last had fresh evidence about each node
                                 (min-plus relayed heartbeat counters, computed
                                 by the protocol layer in iql.py)
        [n_age_slots, ...)      free Q-learned symbols

    That gives the two evaluated detectors: the hand-coded *relay*
    (n_age_slots = N, n_q_slots = 0) and the *learned* channel
    (n_age_slots = 0, n_q_slots = diam(G)). n_age_slots is either 0 or
    number_of_nodes, because every age slot is tied to a specific target node;
    n_q_slots has no such constraint, being free-form learned content with no
    per-target meaning, and is derived as length_message - n_age_slots.

    There is deliberately no hand-coded *belief* broadcast: no evaluated
    detector ever put quantized beliefs on the wire, so the trust gate that
    used to fuse them (a sigmoid score per (sender, target) pair over log-odds,
    age and identity embeddings) never executed in any reported run and has
    been removed. Its parameters survive in checkpoints written earlier;
    load_checkpoint_state drops them, which reloads those runs unchanged
    because forward() never read them.

    Network input per agent = (N-1) sender rows of length_message (with -1 =
    silent/lost/dead, -2 = not a neighbor), flattened, followed by the agent's
    own N-dim age vector. A GRU cell carries evidence across steps so weak
    repeated "dead" votes accumulate instead of being outvoted from scratch
    every step.
    """

    # Parameters that existed only for the removed trust gate. Checkpoints from
    # before the cleanup carry them; forward() never read them.
    LEGACY_PREFIXES = ("node_embedding.", "pair_gate.")
    def __init__(self, number_of_nodes, length_message, message_candidates, seed,
                 q_embed_dim=8, hidden_dim=128, age_cap=15,
                 detach_message_path=True, age_ablation=False,
                 belief_ablation=True, q_stream_only=False, id_embed_dim=None):
        super().__init__()
        # belief_ablation / q_stream_only / id_embed_dim / detach_message_path are
        # accepted only so that checkpoints written before the cleanup still
        # reload: nw_config is saved inside them and replayed as
        # Network(**nw_config). The flags are pinned to the values every reported
        # run used, and anything else is refused rather than silently
        # reinterpreted.
        if not belief_ablation:
            raise ValueError(
                "belief_ablation=False requests the hand-coded belief broadcast, "
                "which has been removed: no evaluated detector broadcast beliefs. "
                "Use the age relay (age_ablation=False) or the learned channel."
            )
        if q_stream_only:
            raise ValueError(
                "q_stream_only has been removed; with the belief broadcast gone the "
                "belief stream is already just the agent's own ages and silence bits."
            )
        if not detach_message_path:
            raise ValueError(
                "detach_message_path=False has been removed: every reported run "
                "detaches the recurrent state on the message path."
            )
        self.number_of_nodes = number_of_nodes
        self.length_message = length_message
        self.message_candidates = message_candidates
        # The message head reads a detached copy of the recurrent state, so
        # encoders/core are trained by the belief loss alone. Kept as an attribute
        # so metrics.json still records it.
        self.detach_message_path = True
        self.age_ablation = bool(age_ablation)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Always 0: the hand-coded belief broadcast is gone. Kept as an attribute
        # so metrics.json still carries the field, and runs from before the
        # cleanup stay field-for-field comparable with runs after it.
        self.n_belief_slots = 0
        # Every age slot is tied to a specific target node, so this is either 0 or
        # number_of_nodes — never a partial size (unlike n_q_slots below).
        self.n_age_slots = 0 if self.age_ablation else self.number_of_nodes
        self.n_q_slots = self.length_message - self.n_age_slots
        if self.n_q_slots < 0:
            raise ValueError(
                f"length_message ({length_message}) must be at least "
                f"n_age_slots ({self.n_age_slots})."
            )
        # n_q_slots == 0 is the relay: no learned message channel at all, only the
        # hand-coded age protocol. The whole message-head / Q pathway
        # is bypassed (see forward / get_actions_batch / max_q below).
        self.q_ablation = self.n_q_slots == 0
        self.n_candidates = len(self.message_candidates)
        self.q_embed_dim = q_embed_dim
        self.hidden_dim = hidden_dim
        self.age_cap = float(age_cap)

        n_senders = self.number_of_nodes - 1
        self.frame_dim = self.length_message
        self.base_input_dim = n_senders * self.frame_dim + self.number_of_nodes

        self.message_q_dim = self.n_q_slots * self.n_candidates

        # Receiver identity is fed to every encoder/head by concatenating this one-hot.
        cond_dim = self.number_of_nodes

        # Sum of learned symbol embeddings across senders. Preserves multi-modal info
        # that a categorical mean-pool would destroy (two distinct symbols ≠ their mean).
        self.q_embedding = nn.Embedding(self.n_candidates, self.q_embed_dim)

        # Per-step hand-coded evidence: the per-target freshest received age (relay
        # only), the agent's own age vector, and per-sender silence bits.
        # See _aggregate for what is actually built.
        if not self.age_ablation:
            belief_stream_dim = 2 * self.number_of_nodes + n_senders
        else:
            belief_stream_dim = self.number_of_nodes + n_senders
        q_stream_dim = self.n_q_slots * self.q_embed_dim

        # Two-stream encoding: each "language" (belief probabilities vs. learned
        # Q-symbols) gets its own encoder so neither MLP has to learn a joint
        # vocabulary. Receiver-conditioned.
        self.belief_encoder = nn.Sequential(
            nn.Linear(belief_stream_dim + cond_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        # Widened from the original 64: under the learned channel this stream is
        # the *only* route for remote liveness, so a narrow bottleneck here would
        # throw away message content the belief stream cannot replace.
        self.q_encoder = nn.Sequential(
            nn.Linear(q_stream_dim + cond_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Recurrent core: replaces the fixed observation window. Hidden state is
        # maintained per agent across the episode by the rollout loop.
        self.core = nn.GRUCell(128 + 128, hidden_dim)

        self.belief_head = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.number_of_nodes),
        )

        # Small trainable adapter on top of the (detached) recurrent state. Lets the
        # message path reshape what it reads — gradient flows here but stops at the
        # detach, so the encoders/core stay trained by belief loss only.
        self.msg_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Message head: deep, residual + pre-norm so the extra depth helps rather than
        # hurts. Kept narrow (128, vs. the old 256) so that growing the number of
        # Q-learned slots doesn't bloat the parameter count — depth (3 residual
        # blocks) buys the extra capacity needed to fill a much larger Q-value
        # table instead of width. Sees its own beliefs + adapted recurrent features.
        msg_in_dim = self.number_of_nodes + hidden_dim + cond_dim
        msg_width = 128
        self.message_head = nn.Sequential(
            nn.Linear(msg_in_dim, msg_width),
            nn.ReLU(),
            ResidualBlock(msg_width),
            ResidualBlock(msg_width),
            ResidualBlock(msg_width),
            nn.LayerNorm(msg_width),
            nn.ReLU(),
            nn.Linear(msg_width, self.message_q_dim),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _aggregate(self, frames, own_age_norm):
        # frames: [B, S, L], own_age_norm: [B, N]
        na = self.n_age_slots

        age_raw = frames[:, :, :na] if na > 0 else None
        q_raw = frames[:, :, na:]

        # Per-sender silence: 1 iff every feature in the row is -1 (dead or lost).
        # Computed from the whole frame, so it stays meaningful whether the row
        # carries age slots or learned slots.
        silent = (frames == -1).all(dim=2).float()  # [B, S]

        # Ages normalized to [0, 1]; missing entries count as maximally stale.
        age_norm = None
        if age_raw is not None:
            age_norm = torch.where(
                age_raw >= 0,
                (age_raw / self.age_cap).clamp(max=1.0),
                torch.ones_like(age_raw),
            )

        if age_norm is not None:
            # Relay: per-target "freshest report seen" summary. This is the whole
            # hand-coded baseline -- receivers keep the smallest reported age per
            # target and hand that vector to the belief head.
            valid_age = (age_raw >= 0).float()
            age_for_min = torch.where(valid_age > 0, age_norm, torch.ones_like(age_norm))
            min_age = age_for_min.min(dim=1).values  # [B, N]
            belief_parts = [min_age]
        else:
            # Learned channel: nothing target-indexed is broadcast at all. Every
            # remote liveness cue has to arrive through the Q-slots below.
            belief_parts = []

        belief_parts.append(own_age_norm)
        belief_parts.append(silent)
        belief_stream = torch.cat(belief_parts, dim=1)

        # Q-slots → learned embeddings, summed over senders and weighted by
        # "did this sender actually speak this step".
        q_idx = q_raw.clamp(min=0).long()
        q_emb = self.q_embedding(q_idx)                      # [B, S, n_q, E]
        valid_q = (q_raw >= 0).float().unsqueeze(-1)
        sender_trust = 1.0 - silent                          # [B, S]
        q_stream = (q_emb * valid_q * sender_trust[:, :, None, None]).sum(dim=1)

        return belief_stream, q_stream.flatten(1)

    def load_checkpoint_state(self, state_dict):
        """Load weights, tolerating checkpoints that predate the cleanup.

        Runs saved before the trust gate was removed carry node_embedding and
        pair_gate tensors. forward() never read them in any reported run (they
        were only reachable through the belief broadcast, which no evaluated
        detector used), so dropping them reproduces those runs exactly. Every
        other key still has to match: this is strict loading with two known-dead
        prefixes filtered out, not a blanket strict=False that would also hide a
        genuinely missing weight.

        Returns the number of legacy tensors dropped.
        """
        kept = {k: v for k, v in state_dict.items()
                if not k.startswith(self.LEGACY_PREFIXES)}
        self.load_state_dict(kept)
        return len(state_dict) - len(kept)

    def _to_id_tensor(self, node_ids):
        return torch.as_tensor(node_ids, dtype=torch.long, device=self.device)

    def forward(self, x, hidden, node_ids, detach_beliefs=False):
        # x: [B, base_input_dim] = flattened sender frames + own age vector.
        ids = self._to_id_tensor(node_ids)
        cond = F.one_hot(ids, num_classes=self.number_of_nodes).float()

        n_senders = self.number_of_nodes - 1
        split = n_senders * self.frame_dim
        frames = x[:, :split].view(x.size(0), n_senders, self.frame_dim)
        own_age_norm = (x[:, split:] / self.age_cap).clamp(0.0, 1.0)

        belief_stream, q_stream = self._aggregate(frames, own_age_norm)

        q_feat = self.q_encoder(torch.cat([q_stream, cond], dim=1))
        b_feat = self.belief_encoder(torch.cat([belief_stream, cond], dim=1))
        h_new = self.core(torch.cat([b_feat, q_feat], dim=1), hidden)

        belief_logits = self.belief_head(torch.cat([h_new, cond], dim=1))
        assert torch.isfinite(belief_logits).all(), "NaN/inf in belief_logits"
        belief_probs = torch.sigmoid(belief_logits)

        if self.q_ablation:
            # Ablation: no learned message channel. Only the supervised belief head
            # is produced; there is no Q-message to return.
            return belief_logits, None, h_new

        belief_for_msg = belief_probs.detach() if detach_beliefs else belief_probs
        # Detach the recurrent state on the message path: encoders + core train via
        # belief loss only. The adapter (trained by message loss) gives the message
        # head trainable capacity over that state without crossing optimizers.
        adapted = self.msg_adapter(h_new.detach())
        q_message = self.message_head(
            torch.cat([belief_for_msg, adapted, cond], dim=1)
        ).view(-1, self.n_q_slots, self.n_candidates)
        return belief_logits, q_message, h_new

    def get_actions_batch(self, batch_data, hidden, node_ids):
        batch = torch.as_tensor(batch_data, dtype=torch.float32, device=self.device)
        h = torch.as_tensor(hidden, dtype=torch.float32, device=self.device)
        ids = self._to_id_tensor(node_ids)
        with torch.no_grad():
            belief_logits, q_message, h_new = self.forward(batch, h, ids)
            belief_probs = torch.sigmoid(belief_logits)
            # An acting node knows it is alive: pin its own slot before the beliefs
            # are scored. Nothing broadcasts them -- see Network's docstring.
            belief_probs[torch.arange(belief_probs.size(0), device=self.device), ids] = 1.0
            if self.q_ablation:
                q_slot_msgs = torch.zeros(
                    (belief_probs.size(0), 0), dtype=torch.long, device=self.device
                )
            else:
                q_slot_msgs = q_message.argmax(dim=2)
        return (
            belief_probs.cpu().numpy(),
            q_slot_msgs.cpu().numpy(),
            h_new.cpu().numpy(),
        )

    def max_q(self, x, hidden, node_ids):
        """Bootstrap value for the message TD target, summed over slots.

        Summing is the structural weakness of this channel rather than an
        implementation detail: all n_q_slots slots share the resulting scalar TD
        error, so per-slot credit is weak (see the paper's Implementation
        Details). It is kept because that is the algorithm the results describe.
        """
        if self.q_ablation:
            return torch.zeros((x.shape[0],), device=self.device)
        h = torch.as_tensor(hidden, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, q_message, _ = self.forward(x, h, node_ids, detach_beliefs=True)
            return q_message.max(dim=2).values.sum(dim=1)
