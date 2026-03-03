from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.sac.configuration_sac import SACConfig


@PreTrainedConfig.register_subclass("sac_stripe")
@dataclass
class SACStripeConfig(SACConfig):
    """SAC + planner-guided BC loss for STRIPE.

    Extends SACConfig with parameters controlling how the planner's
    expert action is used as an auxiliary behavioural-cloning signal
    during actor optimisation.

    New fields:
        planner_bc_weight: Coefficient for the BC loss term added to the
            SAC actor loss.  Set to 0.0 to disable (falls back to vanilla SAC).
        use_planner_q_filter: When True, the BC loss is only applied to
            samples where Q(s, a_planner) > Q(s, a_policy), following the
            Q-filter idea from DDPGfD / DAgger-style methods.
    """

    planner_bc_weight: float = 0.1
    use_planner_q_filter: bool = False
