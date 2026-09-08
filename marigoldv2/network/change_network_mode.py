from marigoldv2.core.registry import REGISTRY


def set_to_eval():
    for k, v in REGISTRY["network_components"].items():
        if hasattr(v, "eval"):
            v.eval()

    # built_network_graphs contains e.g. a graph for training, and could hold a graph for each dataset in case some dataset requires a different logic
    for k, v in REGISTRY["built_network_graphs"].items():
        # A graph contains the forward nodes, i.e. each forward class
        for name, step in v.steps.items():
            if hasattr(step, "set_mode"):
                step.set_mode("test")


def set_to_train():
    for k, v in REGISTRY["network_components"].items():
        if hasattr(v, "train"):
            v.train()

    # built_network_graphs contains e.g. a graph for training, and could hold a graph for each dataset in case some dataset requires a different logic
    for k, v in REGISTRY["built_network_graphs"].items():
        # A graph contains the forward nodes, i.e. each forward class
        for name, step in v.steps.items():
            if hasattr(step, "set_mode"):
                step.set_mode("train")
