import helper as hp
import multiprocessing
from simulations_to_run import SIMULATION_SETS


# ------------------------------------------------------------------------------
FILE_NAME = "results_temp.csv"
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Fixed parameters
TEST_RATIO = 0.20
VCOV_TOLERANCE = 1e-3
MAX_ITER = 2000
ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0]
L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
CV_FOLDS = 10
# ------------------------------------------------------------------------------


def run_simulation_wrapper(params):
    """
    Wrapper function to run a simulation with given parameters.
    """
    # Initialize a unique random seed for each process
    hp.initialize_random_seed()

    # Run the simulation and return the results
    results = hp.run_simulation(
        num_features=params["NUM_FEATURES"],
        covariance_stddev=params["COVARIANCE_STDDEV"],
        covariance_mean=params["COVARIANCE_MEAN"],
        sparsity=params["SPARSITY"],
        signal_to_noise=params["SIGNAL_TO_NOISE"],
        nobs=params["NOBS"],
        test_ratio=TEST_RATIO,
        vcov_tolerance=VCOV_TOLERANCE,
        max_iter=MAX_ITER,
        alphas=ALPHAS,
        l1_ratios=L1_RATIOS,
        cv_folds=CV_FOLDS,
    )
    return results


if __name__ == "__main__":
    # Process each parameter set
    for set_idx, param_set in enumerate(SIMULATION_SETS, 1):
        total_sims = param_set["NUM_BATCHES"] * param_set["THREADS_PER_BATCH"]
        print(f"\nStarting simulation set {set_idx} of {len(SIMULATION_SETS)}")
        print(f"Configuration:")
        print(f"  NUM_FEATURES: {param_set['NUM_FEATURES']}")
        print(f"  NOBS: {param_set['NOBS']:,}")
        print(f"  COVARIANCE_STDDEV: {param_set['COVARIANCE_STDDEV']}")
        print(f"  COVARIANCE_MEAN: {param_set['COVARIANCE_MEAN']}")
        print(f"  SPARSITY: {param_set['SPARSITY']}")
        print(f"  SIGNAL_TO_NOISE: {param_set['SIGNAL_TO_NOISE']}")
        print(f"  NUM_BATCHES: {param_set['NUM_BATCHES']}")
        print(f"  THREADS_PER_BATCH: {param_set['THREADS_PER_BATCH']}")
        print(f"  Total simulations for this set: {total_sims}")

        # Process each batch for this parameter set
        for batch in range(param_set["NUM_BATCHES"]):
            print(f"Running batch {batch + 1} of {param_set['NUM_BATCHES']}...")

            with multiprocessing.Pool() as pool:
                results_list = pool.map(
                    run_simulation_wrapper, [param_set] * param_set["THREADS_PER_BATCH"]
                )

            # Save and upload results after each batch
            hp.save_results_to_csv(results_list, filename=FILE_NAME)
        print(f"Completed all batches for set {set_idx}")

    print("\nAll simulation sets completed!")
