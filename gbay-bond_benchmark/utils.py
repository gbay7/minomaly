from random import choice
from pygod.detector import *
from pyod.models.lof import LOF
from torch_geometric.nn import MLP
from pyod.models.iforest import IForest


def init_model(args):
    dropout = [0.0, 0.1, 0.3]
    lr = [0.1, 0.05, 0.01]
    weight_decay = 0.01

    verbose = 5

    if args.dataset == "inj_flickr" or args.dataset == "dgraph":
        # sampling and minibatch training on large dataset flickr
        batch_size = 64
        num_neigh = 3
        epoch = 2
    else:
        batch_size = 0
        num_neigh = -1
        epoch = 300

    model_name = args.model
    gpu = args.gpu

    if hasattr(args, "epoch"):
        epoch = args.epoch

    if args.dataset == "reddit":
        # for the low feature dimension dataset
        hid_dim = [32, 48, 64]
    elif args.dataset in ["enron", "disney", "dgraph", "books"]:
        hid_dim = [8, 12, 16]
    else:
        hid_dim = [32, 64, 128, 256]

    alpha = [0.8, 0.5, 0.2, 0.1]

    model_settings = {}

    if model_name == "adone":
        model_settings = {
            "hid_dim": choice(hid_dim),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = AdONE(**model_settings)
    elif model_name == "anomalydae":
        hd = choice(hid_dim)
        model_settings = {
            "emb_dim": hd,
            "hid_dim": hd,
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "theta": choice([10.0, 40.0, 90.0]),
            "eta": choice([3.0, 5.0, 8.0]),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            "alpha": choice(alpha),
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = AnomalyDAE(**model_settings)
    elif model_name == "conad":
        model_settings = {
            "hid_dim": choice(hid_dim),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            "weight": choice(alpha),
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = CONAD(**model_settings)
    elif model_name == "dominant":
        model_settings = {
            "hid_dim": choice(hid_dim),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            "weight": choice(alpha),
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = DOMINANT(**model_settings)
    elif model_name == "done":
        model_settings = {
            "hid_dim": choice(hid_dim),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = DONE(**model_settings)
    elif model_name == "gaan":
        model_settings = {
            "noise_dim": choice([8, 16, 32]),
            "hid_dim": choice(hid_dim),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            "weight": choice(alpha),
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = GAAN(**model_settings)
    elif model_name == "gcnae":
        model_settings = {
            "hid_dim": choice(hid_dim),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = GAE(**model_settings)
    elif model_name == "guide":
        model_settings = {
            "a_hid": choice(hid_dim),
            "s_hid": choice([4, 5, 6]),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            "alpha": choice(alpha),
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "cache_dir": "./tmp",
            "verbose": verbose,
        }
        model = GUIDE(**model_settings)
    elif model_name == "mlpae":
        model_settings = {
            "hid_dim": choice(hid_dim),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            "batch_size": batch_size,
            "backbone": MLP,
            "verbose": verbose,
        }
        model = GAE(**model_settings)
    elif model_name == "nepoch_old_lof":
        model_settings = {}
        model = LOF(**model_settings)
    elif model_name == "nepoch_old_if":
        model_settings = {"verbose": verbose}
        model = IForest(**model_settings)
    elif model_name == "radar":
        model_settings = {
            "weight_decay": weight_decay,
            "epoch": epoch,
            "lr": choice(lr),
            "gpu": gpu,
            "verbose": verbose,
        }
        model = Radar(**model_settings)
    elif model_name == "anomalous":
        model_settings = {
            "weight_decay": weight_decay,
            "epoch": epoch,
            "lr": choice(lr),
            "gpu": gpu,
            "verbose": verbose,
        }
        model = ANOMALOUS(**model_settings)
    elif model_name == "nepoch_scan":
        model_settings = {
            "eps": choice([0.3, 0.5, 0.8]),
            "mu": choice([2, 5, 10]),
            "verbose": verbose,
        }
        model = SCAN(**model_settings)


    elif model_name == "cola":
        model_settings = {
            "hid_dim": choice(hid_dim),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            # "weight": choice(alpha),
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = CoLA(**model_settings)
    elif model_name == "dmgd":
        model_settings = {
            "hid_dim": choice(hid_dim),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            # "weight": choice(alpha),
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = DMGD(**model_settings)
    elif model_name == "gadnr":
        model_settings = {
            "hid_dim": choice(hid_dim),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            # "weight": choice(alpha),
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = GADNR(**model_settings)
    elif model_name == "gae":
        model_settings = {
            "hid_dim": choice(hid_dim),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            # "weight": choice(alpha),
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = GAE(**model_settings)
    elif model_name == "ocgnn":
        model_settings = {
            "hid_dim": choice(hid_dim),
            "weight_decay": weight_decay,
            "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            # "weight": choice(alpha),
            "batch_size": batch_size,
            "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = OCGNN(**model_settings)
    elif model_name == "one":
        model_settings = {
            "hid_a": choice(hid_dim),
            "hid_s": choice(hid_dim),
            "weight_decay": weight_decay,
            # "dropout": choice(dropout),
            "lr": choice(lr),
            "epoch": epoch,
            "gpu": gpu,
            "alpha": choice(alpha),
            # "batch_size": batch_size,
            # "num_neigh": num_neigh,
            "verbose": verbose,
        }
        model = ONE(**model_settings)

    return model, model_settings
