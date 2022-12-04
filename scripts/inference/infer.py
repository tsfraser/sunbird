from sunbird.inference import Nested
import time

if __name__ == "__main__":
    nested = Nested.from_abacus_config("configs/infer_tpcf.yaml")
    t0 = time.time()
    print(f"Fitting parameters {nested.param_names}")
    nested()
    print("Fitting took = ", time.time() - t0)
