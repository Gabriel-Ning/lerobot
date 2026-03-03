"""SAC-STRIPE: SAC with planner-guided behavioural-cloning loss.

During the *approach* phase (task_phase == 1) the motion planner produces
an expert action carried alongside every transition (``planner_action``).
``task_phase`` encodes the current phase as an int (1 = approach,
2 = manipulation, 0 = unknown); the BC mask is derived from it at
training time.

This variant adds a weighted MSE (BC) term to the SAC actor loss:

    loss_actor = sac_actor_loss + planner_bc_weight * bc_loss

An optional **Q-filter** restricts the BC term to samples where the
planner action has a higher estimated Q-value than the policy action.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.sac_stripe.configuration_sac_stripe import SACStripeConfig


class SACStripePolicy(SACPolicy):
    """SAC policy augmented with a planner BC loss."""

    config_class = SACStripeConfig
    name = "sac_stripe"

    def __init__(self, config: SACStripeConfig | None = None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # Override forward for the "actor" model to inject BC loss
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic", "temperature", "discrete_critic"] = "critic",
    ) -> dict[str, Tensor]:
        if model != "actor":
            return super().forward(batch, model)

        observations: dict[str, Tensor] = batch["state"]
        observation_features: Tensor | None = batch.get("observation_feature")

        # ---- Standard SAC actor loss (single actor forward) ----
        actions_pi, log_probs, _ = self.actor(observations, observation_features)

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]
        sac_actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()

        # ---- Planner BC loss ----
        bc_loss = torch.tensor(0.0, device=sac_actor_loss.device)

        complementary_info = batch.get("complementary_info")
        if (
            self.config.planner_bc_weight > 0.0
            and complementary_info is not None
            and "planner_action" in complementary_info
        ):
            planner_action = complementary_info["planner_action"]
            task_phase = complementary_info.get("task_phase")

            # Derive planner_valid mask from task_phase (1 = approach → valid)
            if task_phase is not None:
                planner_valid = (task_phase == 1).float().unsqueeze(-1)
            else:
                planner_valid = torch.zeros(
                    planner_action.shape[0], 1, device=planner_action.device,
                )

            if planner_valid.sum() > 0:
                planner_continuous = planner_action[:, : actions_pi.shape[-1]]

                if self.config.use_planner_q_filter:
                    with torch.no_grad():
                        q_planner = self.critic_forward(
                            observations, planner_continuous,
                            use_target=False, observation_features=observation_features,
                        ).min(dim=0)[0]
                        q_policy = self.critic_forward(
                            observations, actions_pi.detach(),
                            use_target=False, observation_features=observation_features,
                        ).min(dim=0)[0]
                        q_filter = (q_planner > q_policy).float().unsqueeze(-1)
                    valid_mask = planner_valid * q_filter
                else:
                    valid_mask = planner_valid

                per_sample = F.mse_loss(actions_pi, planner_continuous, reduction="none")
                bc_loss = (valid_mask * per_sample).sum() / valid_mask.sum().clamp(min=1.0)

        total_loss = sac_actor_loss + self.config.planner_bc_weight * bc_loss

        return {
            "loss_actor": total_loss,
            "loss_bc": bc_loss.detach(),
        }
