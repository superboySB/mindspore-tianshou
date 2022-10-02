"""Policy package."""
# isort:skip_file

from mindrl.policy.base import BasePolicy
from mindrl.policy.random import RandomPolicy
from mindrl.policy.modelfree.dqn import DQNPolicy
from mindrl.policy.modelfree.bdq import BranchingDQNPolicy
from mindrl.policy.modelfree.c51 import C51Policy
from mindrl.policy.modelfree.rainbow import RainbowPolicy
from mindrl.policy.modelfree.qrdqn import QRDQNPolicy
from mindrl.policy.modelfree.iqn import IQNPolicy
from mindrl.policy.modelfree.fqf import FQFPolicy
from mindrl.policy.modelfree.pg import PGPolicy
from mindrl.policy.modelfree.a2c import A2CPolicy
from mindrl.policy.modelfree.npg import NPGPolicy
from mindrl.policy.modelfree.ddpg import DDPGPolicy
from mindrl.policy.modelfree.ppo import PPOPolicy
from mindrl.policy.modelfree.trpo import TRPOPolicy
from mindrl.policy.modelfree.td3 import TD3Policy
from mindrl.policy.modelfree.sac import SACPolicy
from mindrl.policy.modelfree.redq import REDQPolicy
from mindrl.policy.modelfree.discrete_sac import DiscreteSACPolicy
from mindrl.policy.imitation.base import ImitationPolicy
from mindrl.policy.imitation.bcq import BCQPolicy
from mindrl.policy.imitation.cql import CQLPolicy
from mindrl.policy.imitation.td3_bc import TD3BCPolicy
from mindrl.policy.imitation.discrete_bcq import DiscreteBCQPolicy
from mindrl.policy.imitation.discrete_cql import DiscreteCQLPolicy
from mindrl.policy.imitation.discrete_crr import DiscreteCRRPolicy
from mindrl.policy.imitation.gail import GAILPolicy
from mindrl.policy.modelbased.psrl import PSRLPolicy
from mindrl.policy.modelbased.icm import ICMPolicy
from mindrl.policy.multiagent.mapolicy import MultiAgentPolicyManager

__all__ = [
    "BasePolicy",
    "RandomPolicy",
    "DQNPolicy",
    "BranchingDQNPolicy",
    "C51Policy",
    "RainbowPolicy",
    "QRDQNPolicy",
    "IQNPolicy",
    "FQFPolicy",
    "PGPolicy",
    "A2CPolicy",
    "NPGPolicy",
    "DDPGPolicy",
    "PPOPolicy",
    "TRPOPolicy",
    "TD3Policy",
    "SACPolicy",
    "REDQPolicy",
    "DiscreteSACPolicy",
    "ImitationPolicy",
    "BCQPolicy",
    "CQLPolicy",
    "TD3BCPolicy",
    "DiscreteBCQPolicy",
    "DiscreteCQLPolicy",
    "DiscreteCRRPolicy",
    "GAILPolicy",
    "PSRLPolicy",
    "ICMPolicy",
    "MultiAgentPolicyManager",
]
