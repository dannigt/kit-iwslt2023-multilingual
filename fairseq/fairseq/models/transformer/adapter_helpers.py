import logging

logger = logging.getLogger(__name__)

def add_new_layers_to_pretrained(args, pretrained_state, current_state, name):
    """
    Handle loading of a pretrained model, with the purpose of finetuning it
    with (newly-added) adapters / hyper-adapters.
    The (pretrained) state_dict doesn't have any adapter weights, but to load
    the checkpoint we need a 1:1 correspondence between pretrained state_dict
    and the current state_dict. Therefore, to avoid raising errors we  simply
    copy the newly-added weights to the pretrained state_dict and then load.
    Args:
        args: the calling module's arguments
        pretrained_state (dict): state dictionary to upgrade, in place
        current_state (dict): state dictionary of the calling module
        name (str): the state dict key corresponding to the calling module
    Returns:
    """

    for arg, prefix in [("hyper_adapters", "hypernetwork"), ("adapters", "adapters")]:
        # if we are adding adapters / hyper-adapters, but the pretrained checkpoint doesn't have them
        if (getattr(args, arg, False) and
                not any(k.startswith(f"{name}.{prefix}") for k in pretrained_state.keys())):
            for k, v in current_state.items():
                if k.startswith(f"{prefix}."):
                    pretrained_state[f"{name}.{k}"] = v
                    logger.info(f"Added new param '{name}.{k}' to pretrained checkpoint.")

    return pretrained_state