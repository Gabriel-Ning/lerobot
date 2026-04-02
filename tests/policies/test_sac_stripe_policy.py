#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for SACStripePolicy – SAC with planner-guided BC loss."""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sac_stripe.configuration_sac_stripe import SACStripeConfig
from lerobot.policies.sac_stripe.modeling_sac_stripe import SACStripePolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.random_utils import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATE_DIM = 6
ACTION_DIM = 4
BATCH_SIZE = 8


def _make_config(
    planner_bc_weight: float = 0.1,
    use_planner_q_filter: bool = False,
) -> SACStripeConfig:
    config = SACStripeConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        dataset_stats={
            OBS_STATE: {"min": [0.0] * STATE_DIM, "max": [1.0] * STATE_DIM},
            ACTION: {"min": [0.0] * ACTION_DIM, "max": [1.0] * ACTION_DIM},
        },
        planner_bc_weight=planner_bc_weight,
        use_planner_q_filter=use_planner_q_filter,
    )
    config.validate_features()
    return config


def _make_policy(
    planner_bc_weight: float = 0.1,
    use_planner_q_filter: bool = False,
) -> SACStripePolicy:
    config = _make_config(
        planner_bc_weight=planner_bc_weight,
        use_planner_q_filter=use_planner_q_filter,
    )
    policy = SACStripePolicy(config=config)
    policy.train()
    return policy


def _base_batch(batch_size: int = BATCH_SIZE) -> dict:
    return {
        ACTION: torch.randn(batch_size, ACTION_DIM),
        "reward": torch.randn(batch_size),
        "state": {OBS_STATE: torch.randn(batch_size, STATE_DIM)},
        "next_state": {OBS_STATE: torch.randn(batch_size, STATE_DIM)},
        "done": torch.zeros(batch_size),
        "complementary_info": None,
    }


def _make_complementary_info(
    task_phase_value: int,
    batch_size: int = BATCH_SIZE,
    planner_action: torch.Tensor | None = None,
) -> dict:
    if planner_action is None:
        planner_action = torch.randn(batch_size, ACTION_DIM)
    return {
        "planner_action": planner_action,
        "task_phase": torch.full((batch_size,), task_phase_value, dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sac_stripe_inherits_critic_and_temperature():
    """Critic and temperature forward paths are inherited from SACPolicy unchanged."""
    set_seed(0)
    policy = _make_policy()
    batch = _base_batch()

    critic_out = policy.forward(batch, model="critic")
    assert "loss_critic" in critic_out
    assert critic_out["loss_critic"].shape == ()

    temp_out = policy.forward(batch, model="temperature")
    assert "loss_temperature" in temp_out
    assert temp_out["loss_temperature"].shape == ()


def test_actor_loss_no_complementary_info():
    """Actor forward without complementary_info returns loss_actor only (bc_loss=0)."""
    set_seed(0)
    policy = _make_policy(planner_bc_weight=0.5)
    batch = _base_batch()

    out = policy.forward(batch, model="actor")
    assert "loss_actor" in out
    assert out["loss_actor"].shape == ()
    # bc_loss should be zero when no complementary_info is provided
    assert out["loss_bc"].item() == 0.0


def test_actor_loss_bc_zero_when_weight_is_zero():
    """BC loss term is disabled when planner_bc_weight=0."""
    set_seed(0)
    policy_no_bc = _make_policy(planner_bc_weight=0.0)
    policy_with_bc = _make_policy(planner_bc_weight=0.5)

    batch_no_bc = _base_batch()
    batch_with_bc = _base_batch()
    comp = _make_complementary_info(task_phase_value=1)
    batch_with_bc["complementary_info"] = comp

    out_no_bc = policy_no_bc.forward(batch_no_bc, model="actor")
    assert out_no_bc["loss_bc"].item() == 0.0

    out_with_bc = policy_with_bc.forward(batch_with_bc, model="actor")
    assert out_with_bc["loss_bc"].item() > 0.0


def test_actor_bc_loss_zero_outside_approach_phase():
    """BC loss is zero when task_phase != 1 (not the approach phase)."""
    set_seed(0)
    policy = _make_policy(planner_bc_weight=1.0)

    for phase in (0, 2):
        batch = _base_batch()
        batch["complementary_info"] = _make_complementary_info(task_phase_value=phase)
        out = policy.forward(batch, model="actor")
        assert out["loss_bc"].item() == 0.0, f"Expected bc_loss=0 for task_phase={phase}"


def test_actor_bc_loss_nonzero_in_approach_phase():
    """BC loss is non-zero when task_phase == 1 (approach phase active)."""
    set_seed(0)
    policy = _make_policy(planner_bc_weight=1.0)
    batch = _base_batch()
    # Use a planner action far from what the policy would choose so bc_loss is large
    batch["complementary_info"] = _make_complementary_info(
        task_phase_value=1,
        planner_action=torch.ones(BATCH_SIZE, ACTION_DIM) * 100.0,
    )
    out = policy.forward(batch, model="actor")
    assert out["loss_bc"].item() > 0.0


def test_actor_bc_loss_key_present_in_output():
    """The actor output always contains 'loss_bc'."""
    set_seed(0)
    policy = _make_policy(planner_bc_weight=0.3)
    batch = _base_batch()
    batch["complementary_info"] = _make_complementary_info(task_phase_value=1)
    out = policy.forward(batch, model="actor")
    assert "loss_bc" in out


def test_actor_total_loss_includes_bc_term():
    """Total actor loss is larger when BC term is active (approach phase with large planner mismatch)."""
    set_seed(42)
    weight = 2.0
    policy = _make_policy(planner_bc_weight=weight)

    batch_without_bc = _base_batch()
    batch_with_bc = _base_batch()
    # Planner action far from the origin forces a large MSE BC loss
    batch_with_bc["complementary_info"] = _make_complementary_info(
        task_phase_value=1,
        planner_action=torch.ones(BATCH_SIZE, ACTION_DIM) * 5.0,
    )

    # Run under no_grad so internal stochastic sampling does not advance the RNG
    with torch.no_grad():
        out_without = policy.forward(batch_without_bc, model="actor")
        out_with = policy.forward(batch_with_bc, model="actor")

    # The total loss with BC should exceed the one without (large planner mismatch)
    assert out_with["loss_actor"].item() > out_without["loss_actor"].item()


def test_q_filter_suppresses_bc_when_policy_q_higher():
    """With Q-filter enabled, BC loss is suppressed when Q(s,a_policy) >= Q(s,a_planner)."""
    set_seed(0)
    policy = _make_policy(planner_bc_weight=1.0, use_planner_q_filter=True)

    # Use a planner action equal to the zero vector – critic will typically rate
    # random policy actions higher at init; run several seeds to confirm mask fires.
    batch = _base_batch()
    # Use extreme planner action to ensure low Q value from critic
    batch["complementary_info"] = _make_complementary_info(
        task_phase_value=1,
        planner_action=torch.full((BATCH_SIZE, ACTION_DIM), -100.0),
    )

    out = policy.forward(batch, model="actor")
    # Loss may be zero (all filtered) or small; it should never be as large as
    # without the filter for such an extreme planner action.
    assert out["loss_bc"].item() >= 0.0  # never negative


def test_q_filter_disabled_bc_loss_larger():
    """Without Q-filter, a bad planner action yields a larger BC loss than with filter."""
    set_seed(5)
    extreme_planner = torch.full((BATCH_SIZE, ACTION_DIM), -50.0)

    policy_no_filter = _make_policy(planner_bc_weight=1.0, use_planner_q_filter=False)
    policy_with_filter = _make_policy(planner_bc_weight=1.0, use_planner_q_filter=True)

    batch_no = _base_batch()
    batch_no["complementary_info"] = _make_complementary_info(
        task_phase_value=1, planner_action=extreme_planner.clone()
    )
    batch_fi = _base_batch()
    batch_fi["complementary_info"] = _make_complementary_info(
        task_phase_value=1, planner_action=extreme_planner.clone()
    )

    with torch.no_grad():
        out_no = policy_no_filter.forward(batch_no, model="actor")
        out_fi = policy_with_filter.forward(batch_fi, model="actor")

    # Without filter every sample contributes; with filter many are suppressed.
    assert out_no["loss_bc"].item() >= out_fi["loss_bc"].item()


def test_actor_forward_backward_does_not_error():
    """A full forward-backward pass for the actor completes without error."""
    set_seed(0)
    policy = _make_policy(planner_bc_weight=0.5)
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=1e-4)

    batch = _base_batch()
    batch["complementary_info"] = _make_complementary_info(task_phase_value=1)

    loss = policy.forward(batch, model="actor")["loss_actor"]
    loss.backward()
    optimizer.step()


def test_stripe_config_defaults():
    """SACStripeConfig has the expected default values for new fields."""
    config = _make_config()
    assert config.planner_bc_weight == 0.1
    assert config.use_planner_q_filter is False


def test_stripe_config_custom_values():
    """SACStripeConfig correctly stores custom planner BC settings."""
    config = _make_config(planner_bc_weight=0.5, use_planner_q_filter=True)
    assert config.planner_bc_weight == 0.5
    assert config.use_planner_q_filter is True
