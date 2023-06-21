import argparse
import yaml
import time
from pathlib import Path
from sunbird.inference import Nested

if __name__ == "__main__":
    output_path = Path("/pscratch/sd/t/tsfraser/sunbird/chains/tsfraser/")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="infer_new.yaml"
    )
    # Make sure it reads from dataset with fixed hod
    parser.add_argument("--cosmology", type=int, default=0)
    parser.add_argument("--hod_idx", type=int, default=26)
    parser.add_argument("--suffix", type=str, default=None)
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config["data"]["observation"]["get_obs_args"]["cosmology"] = args.cosmology
    config["data"]["observation"]["get_obs_args"]["hod_idx"] = args.hod_idx
    loss = config["theory_model"]["args"]["loss"]
    vol = config["data"]["covariance"]["volume_scaling"]
    smin = config["slice_filters"]["s"][0]
    smax = config["slice_filters"]["s"][1]
    statistics = "_".join([i for i in config["statistics"]])
    multipoles = ''.join([str(i) for i in config["select_filters"]["multipoles"]])
    #quintiles = ''.join([str(i) for i in config["select_filters"]["quintiles"]])
    dir_store = f"abacus_cosmo{args.cosmology}_hod{args.hod_idx}_{statistics}_"\
                f"{loss}_vol{vol}_smin{smin:.2f}_smax{smax:.2f}_m{multipoles}"
    if args.suffix is not None:
        dir_store += f"_{args.suffix}"
    config["inference"]["output_dir"] = output_path / dir_store
    print("output dir")
    print(config["inference"]["output_dir"])
    nested = Nested.from_config_dict(
        config=config,
    )
    t0 = time.time()
    print(f"Fitting parameters {nested.param_names}")
    nested()
    print("Fitting took = ", time.time() - t0)
