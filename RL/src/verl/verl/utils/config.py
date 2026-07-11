# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import is_dataclass
from typing import Any, Optional
import warnings

from omegaconf import DictConfig, ListConfig, OmegaConf

__all__ = ["omega_conf_to_dataclass", "validate_config"]


def _coerce_cfg_int(value: Any, field_name: str) -> int:
    """Coerce config values to int for arithmetic validation paths."""
    if isinstance(value, bool):
        raise TypeError(f"Config field '{field_name}' must be an integer, got bool: {value!r}")
    try:
        return int(value)
    except Exception as exc:
        raise TypeError(
            f"Config field '{field_name}' must be an integer-compatible value, got {type(value).__name__}: {value!r}"
        ) from exc


def _align_ppo_mini_batch_size_to_train_batch_size(
    config: DictConfig,
    *,
    section_path: str,
    train_batch_size: int,
) -> None:
    """Keep PPO mini-batch in prompt-batch units aligned with data.train_batch_size.

    In this codebase, `actor/critic.ppo_mini_batch_size` is configured in the same
    per-step prompt-batch unit as `data.train_batch_size`, while rollout updates
    later expand it by `rollout.n`. If a launch only overrides `data.train_batch_size`
    and leaves a larger stale `ppo_mini_batch_size` behind, config validation fails
    before training starts even though the intent is an all-in-one update.
    """
    section = OmegaConf.select(config, section_path)
    if section is None:
        return

    raw_value = OmegaConf.select(config, f"{section_path}.ppo_mini_batch_size")
    if raw_value in (None, ""):
        return

    mini_batch_size = _coerce_cfg_int(raw_value, f"{section_path}.ppo_mini_batch_size")
    if mini_batch_size <= train_batch_size:
        setattr(section, "ppo_mini_batch_size", mini_batch_size)
        return

    warnings.warn(
        f"Auto-aligning {section_path}.ppo_mini_batch_size from {mini_batch_size} to "
        f"data.train_batch_size ({train_batch_size}) so small-batch overrides remain valid. "
        f"Override {section_path}.ppo_mini_batch_size explicitly if a different value is intended.",
        stacklevel=2,
    )
    setattr(section, "ppo_mini_batch_size", train_batch_size)


def _enable_remove_padding_for_sequence_parallelism(config: DictConfig, *, use_critic: bool) -> None:
    actor_strategy = str(OmegaConf.select(config, "actor_rollout_ref.actor.strategy") or "").strip().lower()
    actor_sp_size = _coerce_cfg_int(
        OmegaConf.select(config, "actor_rollout_ref.actor.ulysses_sequence_parallel_size") or 1,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size",
    )
    if actor_strategy in {"fsdp", "fsdp2"} and actor_sp_size > 1:
        actor_model = OmegaConf.select(config, "actor_rollout_ref.model")
        actor_use_remove_padding = bool(
            OmegaConf.select(config, "actor_rollout_ref.model.use_remove_padding") or False
        )
        if actor_model is not None and not actor_use_remove_padding:
            warnings.warn(
                "Auto-enabling actor_rollout_ref.model.use_remove_padding because "
                "actor_rollout_ref.actor.ulysses_sequence_parallel_size > 1 requires remove-padding.",
                stacklevel=2,
            )
            setattr(actor_model, "use_remove_padding", True)

    if not use_critic:
        return

    critic_strategy = str(OmegaConf.select(config, "critic.strategy") or "").strip().lower()
    critic_sp_size = _coerce_cfg_int(
        OmegaConf.select(config, "critic.ulysses_sequence_parallel_size") or 1,
        "critic.ulysses_sequence_parallel_size",
    )
    if critic_strategy in {"fsdp", "fsdp2"} and critic_sp_size > 1:
        critic_model = OmegaConf.select(config, "critic.model")
        critic_use_remove_padding = bool(OmegaConf.select(config, "critic.model.use_remove_padding") or False)
        if critic_model is not None and not critic_use_remove_padding:
            warnings.warn(
                "Auto-enabling critic.model.use_remove_padding because "
                "critic.ulysses_sequence_parallel_size > 1 requires remove-padding.",
                stacklevel=2,
            )
            setattr(critic_model, "use_remove_padding", True)


def omega_conf_to_dataclass(config: DictConfig | dict, dataclass_type: Optional[type[Any]] = None) -> Any:
    """
    Convert an OmegaConf DictConfig to a dataclass.

    Args:
        config: The OmegaConf DictConfig or dict to convert.
        dataclass_type: The dataclass type to convert to. When dataclass_type is None,
            the DictConfig must contain _target_ to be instantiated via hydra.instantiate API.

    Returns:
        The dataclass instance.
    """
    # Got an empty config
    if not config:
        return dataclass_type if dataclass_type is None else dataclass_type()
    # Got an object
    if not isinstance(config, DictConfig | ListConfig | dict | list):
        return config

    if dataclass_type is None:
        assert "_target_" in config, (
            "When dataclass_type is not provided, config must contain _target_. "
            "See trainer/config/ppo_trainer.yaml algorithm section for an example. "
            f"Got config: {config}"
        )
        from hydra.utils import instantiate

        return instantiate(config, _convert_="partial")

    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} must be a dataclass")
    cfg = OmegaConf.create(config)  # in case it's a dict
    # pop _target_ to avoid hydra instantiate error, as most dataclass do not have _target_
    # Updated (vermouth1992) We add _target_ to BaseConfig so that it is compatible.
    # Otherwise, this code path can't support recursive instantiation.
    # if "_target_" in cfg:
    #     cfg.pop("_target_")
    cfg_from_dataclass = OmegaConf.structured(dataclass_type)
    # let cfg override the existing vals in `cfg_from_dataclass`
    cfg_merged = OmegaConf.merge(cfg_from_dataclass, cfg)
    # now convert to `dataclass_type`
    config_object = OmegaConf.to_object(cfg_merged)
    return config_object


def update_dict_with_config(dictionary: dict, config: DictConfig):
    for key in dictionary:
        if hasattr(config, key):
            dictionary[key] = getattr(config, key)


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """
    n_gpus_per_node = _coerce_cfg_int(config.trainer.n_gpus_per_node, "trainer.n_gpus_per_node")
    nnodes = _coerce_cfg_int(config.trainer.nnodes, "trainer.nnodes")
    train_batch_size = _coerce_cfg_int(config.data.train_batch_size, "data.train_batch_size")
    rollout_n = _coerce_cfg_int(config.actor_rollout_ref.rollout.n, "actor_rollout_ref.rollout.n")

    # Normalize key numeric fields to ints so downstream runtime code paths don't receive oc.env strings.
    config.trainer.n_gpus_per_node = n_gpus_per_node
    config.trainer.nnodes = nnodes
    config.data.train_batch_size = train_batch_size
    config.actor_rollout_ref.rollout.n = rollout_n

    # number of GPUs total
    n_gpus = n_gpus_per_node * nnodes

    effective_rollout_batch_size = train_batch_size * rollout_n

    _align_ppo_mini_batch_size_to_train_batch_size(
        config,
        section_path="actor_rollout_ref.actor",
        train_batch_size=train_batch_size,
    )
    if use_critic:
        _align_ppo_mini_batch_size_to_train_batch_size(
            config,
            section_path="critic",
            train_batch_size=train_batch_size,
        )
    _enable_remove_padding_for_sequence_parallelism(config, use_critic=use_critic)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if config.actor_rollout_ref.actor.strategy == "megatron":
            megatron_tp = _coerce_cfg_int(
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size,
                "actor_rollout_ref.actor.megatron.tensor_model_parallel_size",
            )
            megatron_pp = _coerce_cfg_int(
                config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size,
                "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size",
            )
            megatron_cp = _coerce_cfg_int(
                config.actor_rollout_ref.actor.megatron.context_parallel_size,
                "actor_rollout_ref.actor.megatron.context_parallel_size",
            )
            actor_micro_batch_size = _coerce_cfg_int(
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
            )

            config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size = megatron_tp
            config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size = megatron_pp
            config.actor_rollout_ref.actor.megatron.context_parallel_size = megatron_cp
            config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = actor_micro_batch_size
            model_parallel_size = (
                megatron_tp
                * megatron_pp
            )
            assert n_gpus % (model_parallel_size * megatron_cp) == 0, (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({megatron_cp})"
            )
            megatron_dp = n_gpus // (model_parallel_size * megatron_cp)
            minimal_bsz = megatron_dp * actor_micro_batch_size
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = effective_rollout_batch_size
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
    # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    # Actor validation done in ActorConfig.__post_init__ and validate()
    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    actor_config.validate(n_gpus, effective_rollout_batch_size, config.actor_rollout_ref.model)

    if not config.actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref",
            )

        #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
        check_mutually_exclusive(
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "actor_rollout_ref.rollout",
        )

    if config.algorithm.get("use_kl_in_reward", False) and config.actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(n_gpus, effective_rollout_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    lora_config = config.actor_rollout_ref.model.get("lora", {})
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank > 0 and config.actor_rollout_ref.rollout.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    print("[validate_config] All configuration checks passed successfully!")
