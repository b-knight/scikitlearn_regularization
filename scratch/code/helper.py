import warnings
import csv
import random
import numpy as np
import pandas as pd
import snowflake.connector
from scipy.stats import norm
from scipy.linalg import eigh
from scipy.stats import truncnorm
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from snowflake.connector.pandas_tools import write_pandas
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LassoLarsCV
import gc
import os

warnings.filterwarnings("ignore")


def nearest_correlation_matrix(A, tol=1e-8, max_iter=100):
    """
    Uses Higham's algorithm to find the nearest correlation matrix to A.
    A must be symmetric.
    """
    Y = A.copy()
    delta_S = np.zeros_like(A)
    for _ in range(max_iter):
        R = Y - delta_S
        # Project onto the positive semidefinite cone:
        eigvals, eigvecs = eigh(R)
        eigvals[eigvals < 0] = 0
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        delta_S = X - R
        # Force unit diagonal:
        Y_new = X.copy()
        np.fill_diagonal(Y_new, 1)
        if np.linalg.norm(Y_new - Y, ord="fro") < tol:
            return Y_new
        Y = Y_new
    return Y


def ensure_strict_pd(matrix, eps=1e-8):
    """
    Ensure that a matrix is strictly positive definite by adding jitter if necessary.

    Parameters:
        matrix (np.ndarray): A symmetric, PSD matrix.
        eps (float): A small constant to add to the diagonal if needed.

    Returns:
        np.ndarray: A strictly positive definite matrix.
    """
    # Get the minimum eigenvalue
    eigvals = np.linalg.eigvalsh(matrix)
    min_eig = np.min(eigvals)
    if min_eig <= 0:
        matrix += np.eye(matrix.shape[0]) * (abs(min_eig) + eps)
    return matrix


def generate_corr_matrix(n, off_diag_std, off_diag_mean=0.0):
    """
    Generates an n x n correlation matrix with a specified off-diagonal
    mean and standard deviation (approximately), then projects it to a
    valid correlation matrix.

    Args:
        n (int): Size of the matrix.
        off_diag_std (float): Target std deviation of off-diagonal elements.
        off_diag_mean (float): Target mean of off-diagonal elements.

    Returns:
        np.ndarray: A valid correlation matrix.
    """
    # Ensure standard deviation is positive
    off_diag_std = max(off_diag_std, 1e-8)
    lower_bound, upper_bound = -1, 1
    a, b = (lower_bound - off_diag_mean) / off_diag_std, (
        upper_bound - off_diag_mean
    ) / off_diag_std

    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            value = truncnorm.rvs(a, b, loc=off_diag_mean, scale=off_diag_std)
            A[i, j] = value
            A[j, i] = value
    np.fill_diagonal(A, 1)
    corr = nearest_correlation_matrix(A)
    return ensure_strict_pd(corr)


def extract_offdiagonals(matrix):
    """
    Extracts the off-diagonal elements from a symmetric matrix.
    """
    triu_indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[triu_indices]


def confirm_covariance_std(matrix, ddof=0):
    """
    Computes the standard deviation of the off-diagonal elements.
    """
    off_diag_values = extract_offdiagonals(matrix)
    return np.std(off_diag_values, ddof=ddof)


def calibrate_corr_matrix(
    n, target_std, target_mean=0.0, tol=1e-3, max_iter=5000, initial_guess=None
):
    """
    Calibrates the input off-diagonal standard deviation parameter so that the
    final correlation matrix has off-diagonals with a standard deviation close
    to target_std.

    Returns:
        tuple: (calibrated correlation matrix, input std used, final off-diagonal std)
    """
    start_time = datetime.now()
    # print(f"🔧 Starting correlation matrix calibration ({n}x{n}, target_std={target_std:.3f}, target_mean={target_mean:.3f})")

    if initial_guess is None:
        off_diag_std_in = target_std * 1.5
    else:
        off_diag_std_in = initial_guess

    for i in range(max_iter):
        # Ensure off_diag_std_in is positive
        off_diag_std_in = max(off_diag_std_in, 1e-6)

        corr_matrix = generate_corr_matrix(
            n, off_diag_std_in, off_diag_mean=target_mean
        )
        actual_std = confirm_covariance_std(corr_matrix)
        error = actual_std - target_std
        if abs(error) < tol:
            actual_mean = np.mean(extract_offdiagonals(corr_matrix))
            duration = (datetime.now() - start_time).total_seconds()
            # print(f"✅ Correlation matrix calibration complete in {duration:.2f}s (converged after {i+1} iterations)")
            return corr_matrix, off_diag_std_in, actual_std, target_mean, actual_mean
        if actual_std == 0:
            off_diag_std_in *= 2
        else:
            off_diag_std_in *= target_std / actual_std

        # Ensure off_diag_std_in remains positive after update
        off_diag_std_in = max(off_diag_std_in, 1e-6)

    raise ValueError("Calibration failed to converge within the maximum iterations.")


def simulate_correlated_coefficients(Sigma, sparsity=0.5):
    """
    Simulate beta coefficients with a given covariance structure (Sigma),
    then transform the marginals from normal to Laplace (double exponential)
    via a copula transformation.

    Args:
        p (int): Number of coefficients.
        Sigma (np.ndarray): p x p covariance matrix.

    Returns:
        beta_normal (np.ndarray): The original multivariate normal sample.
        beta_laplace (np.ndarray): The transformed sample with Laplace marginals.
    """
    start_time = datetime.now()
    p = Sigma.shape[0]
    # print(f"🎯 Generating {p} correlated coefficients (sparsity={sparsity:.2f})")

    # Step 1: Draw a multivariate normal sample with mean 0
    beta_normal = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma)

    # Step 2: Transform each component to Uniform(0,1) using the normal CDF
    u = norm.cdf(beta_normal)

    # Step 3: Transform the uniform variates to Laplace using the inverse CDF
    # Inverse CDF for Laplace(0,1):
    #   if u < 0.5, then x = ln(2u)
    #   if u >= 0.5, then x = -ln[2(1-u)]
    beta_laplace = np.where(u < 0.5, np.log(2 * u), -np.log(2 * (1 - u)))

    # Step 4: Thresholding to introduce sparsity
    threshold = np.quantile(np.abs(beta_laplace), sparsity)
    beta = beta_laplace * (np.abs(beta_laplace) > threshold)

    duration = (datetime.now() - start_time).total_seconds()
    non_zero_count = np.count_nonzero(beta)
    # print(f"✅ Coefficient generation complete in {duration:.3f}s ({non_zero_count}/{p} non-zero coefficients)")

    return beta


def gen_correlated_data(corr_matrix, nobs, mean=None):
    """
    Generate correlated data from a given correlation (or covariance) matrix.

    Args:
        corr_matrix (np.ndarray): p x p correlation or covariance matrix.
        nobs (int): Number of observations to generate.
        mean (np.ndarray or list, optional): Mean vector of length p.
                                               Defaults to a zero vector.

    Returns:
        pd.DataFrame: DataFrame with nobs rows and p columns representing the correlated data.
    """
    start_time = datetime.now()
    p = corr_matrix.shape[0]
    # print(f"📊 Generating {nobs:,} observations with {p} correlated features")
    # Perform the Cholesky decomposition
    L = np.linalg.cholesky(corr_matrix)

    # Generate independent standard normal data
    uncorrelated = np.random.standard_normal((nobs, p))

    # Apply the Cholesky factor to introduce correlations
    correlated = np.dot(uncorrelated, L.T)

    # Add a mean if provided (default is zero)
    if mean is not None:
        mean = np.array(mean)
        if mean.shape[0] != p:
            raise ValueError(
                "Mean vector must have the same length as the number of variables."
            )
        correlated += mean  # This adds the mean to each row

    # Return the data as a DataFrame (rows: observations, columns: variables)
    duration = (datetime.now() - start_time).total_seconds()
    # print(f"✅ Data generation complete in {duration:.2f}s ({nobs:,} x {p} matrix)")

    return pd.DataFrame(correlated)


def generate_response(X, beta, signal_to_noise_ratio=1.0):
    """
    Generate response variable Y from X and beta, adding Gaussian noise.

    Args:
    - X (np.ndarray): Feature matrix (n x p).
    - beta (np.ndarray): Coefficient vector (p,).
    - noise_level (float): Controls noise variance relative to signal variance.

    Returns:
    - Y (np.ndarray): Response vector (n,).
    """
    start_time = datetime.now()
    # print(f"🎲 Generating response variable (SNR={signal_to_noise_ratio:.2f})")

    signal = X @ beta
    signal_variance = np.var(signal)
    noise_variance = (
        signal_variance / signal_to_noise_ratio
    )  # Control noise relative to signal strength
    noise = np.random.normal(0, np.sqrt(noise_variance), size=X.shape[0])

    duration = (datetime.now() - start_time).total_seconds()
    y = signal + noise
    # print(f"✅ Response generation complete in {duration:.3f}s (mean={np.mean(y):.2f}, std={np.std(y):.2f})")

    return y


def create_results_table(snowflake_connection):
    table_creation_query = """
    CREATE TABLE IF NOT EXISTS SANDBOX_DB.BENJAMINKNIGHT.REGULARIZATION_SIMULATION_RESULTS (
    NUM_FEATURES NUMBER(38,0),
    NOBS NUMBER(38,0),
    COVARIANCE_STDDEV NUMBER(38,8),
    SPARSITY NUMBER(38,8),
    SIGNAL_TO_NOISE NUMBER(38,8),
    VCOV_TOLERANCE VARCHAR,
    MAX_ITER NUMBER(38,0),
    ALPHAS VARCHAR,
    L1_RATIOS VARCHAR,
    CV_FOLDS NUMBER(38,0),
    TEST_RATIO NUMBER(38,12),
    LASSO_RMSE NUMBER(38,12),
    RIDGE_RMSE NUMBER(38,12),
    ELASTIC_NET_RMSE NUMBER(38,12),
    LASSO_DERIVATION_SEC NUMBER(38,0),
    RIDGE_DERIVATION_SEC NUMBER(38,0),
    ELASTIC_NET_DERIVATION_SEC NUMBER(38,0),
    Y_MEAN NUMBER(38,8),
    Y_STDDEV NUMBER(38,8),
    CREATED_AT_UTC TIMESTAMP_NTZ(9),
    PARAMETERS_USED VARCHAR(500),
    COVARIANCE_MEAN NUMBER(38,8)
    )  
    """
    snowflake_connection.cursor().execute(table_creation_query)
    print(
        "Created the table SANDBOX_DB.BENJAMINKNIGHT.REGULARIZATION_SIMULATION_RESULTS."
    )


def initialize_random_seed():
    """
    Initialize a unique random seed for each process.
    Uses process ID (PID) to ensure randomness across different processes.
    """
    seed = os.getpid()  # Get the process ID
    np.random.seed(seed)
    random.seed(seed)


def run_regularization_sims(
    data,
    covariance_std,
    covariance_mean,
    sparsity,
    test_ratio,
    snr,
    vcov_std_tol,
    max_iters,
    allowed_alphas,
    allowed_l1_ratios,
    cv_folds,
):
    """
    Run regularization simulations using Lasso, Ridge, ElasticNet, and LARS models.

    This function splits the input data into training and testing sets, fits the specified
    regularization models, and calculates the root mean square error (RMSE) for each model.
    It also records the time taken to derive each model.

    Args:
        data (pd.DataFrame): The input data containing features and the target variable 'Y'.
        covariance_std (float): The standard deviation of the covariance.
        sparsity (float): The sparsity level of the coefficients.
        test_ratio (float): The proportion of the dataset to include in the test split.
        snr (float): The signal-to-noise ratio.
        vcov_std_tol (float): The tolerance for the variance-covariance standard deviation.
        max_iters (int): The maximum number of iterations for model fitting.
        allowed_alphas (list): List of alpha values to consider for Lasso and ElasticNet.
        allowed_l1_ratios (list): List of L1 ratios to consider for ElasticNet.
        cv_folds (int): The number of cross-validation folds.

    Returns:
        dict: A dictionary containing the RMSE and derivation time for each model,
              along with other simulation parameters.
    """
    # print(f"🤖 Starting regularization model fitting ({cv_folds}-fold CV)")
    overall_start = datetime.now()

    results_dict = {}
    X, y = data[data.columns[:-1]], data["Y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=1
    )

    # print(f"   Training set: {X_train.shape[0]:,} samples x {X_train.shape[1]} features")
    # print(f"   Test set: {X_test.shape[0]:,} samples")

    # Fit Lasso
    # print("   🔵 Fitting Lasso with cross-validation...")
    start_time = datetime.now()
    lasso_model = LassoCV(
        alphas=allowed_alphas, cv=cv_folds, random_state=0, n_jobs=-1
    ).fit(X_train, y_train)
    end_time = datetime.now()
    lasso_time = (end_time - start_time).seconds
    results_dict["lasso_derivation_time"] = lasso_time
    # print(f"      ✅ Lasso complete in {lasso_time}s (α={lasso_model.alpha_:.4f})")

    # Fit Ridge
    # print("   🟠 Fitting Ridge with cross-validation...")
    start_time = datetime.now()
    ridge_model = RidgeCV(alphas=allowed_alphas, cv=cv_folds).fit(X_train, y_train)
    end_time = datetime.now()
    ridge_time = (end_time - start_time).seconds
    results_dict["ridge_derivation_time"] = ridge_time
    # print(f"      ✅ Ridge complete in {ridge_time}s (α={ridge_model.alpha_:.4f})")

    # Fit ElasticNet
    # print("   🟢 Fitting ElasticNet with cross-validation...")
    start_time = datetime.now()
    elastic_model = ElasticNetCV(
        alphas=allowed_alphas,
        cv=cv_folds,
        random_state=0,
        l1_ratio=allowed_l1_ratios,
        n_jobs=-1,
    ).fit(X_train, y_train)
    end_time = datetime.now()
    elastic_time = (end_time - start_time).seconds
    results_dict["elastc_derivation_time"] = elastic_time
    # print(f"      ✅ ElasticNet complete in {elastic_time}s (α={elastic_model.alpha_:.4f}, l1_ratio={elastic_model.l1_ratio_:.2f})")

    lasso_predictions = lasso_model.predict(X_test)
    ridge_predictions = ridge_model.predict(X_test)
    elastic_predictions = elastic_model.predict(X_test)
    pred_df = pd.DataFrame(
        [
            lasso_predictions,
            ridge_predictions,
            elastic_predictions,
            y_test,
        ]
    ).T
    pred_df.columns = [
        "Lasso_Pred",
        "Ridge_Pred",
        "Elastic_Pred",
        "Y",
    ]
    pred_df["Lasso_Difference"] = (pred_df["Lasso_Pred"] - pred_df["Y"]) ** 2
    pred_df["Ridge_Difference"] = (pred_df["Ridge_Pred"] - pred_df["Y"]) ** 2
    pred_df["Elastic_Difference"] = (pred_df["Elastic_Pred"] - pred_df["Y"]) ** 2
    results_dict["num_features"] = len(data.columns) - 1
    results_dict["nobs"] = len(data)
    results_dict["covariance_std"] = covariance_std
    results_dict["covariance_mean"] = covariance_mean
    results_dict["sparsity"] = sparsity
    results_dict["test_ratio"] = test_ratio
    results_dict["signal_to_noise"] = snr
    results_dict["vcov_std_tol"] = str(vcov_std_tol)
    results_dict["max_iterations"] = max_iters
    results_dict["candidate_alpha_values"] = allowed_alphas
    results_dict["candidate_l1_ratios"] = allowed_l1_ratios
    results_dict["cv_folds_used"] = cv_folds
    results_dict["lasso_rmse"] = pred_df["Lasso_Difference"].mean() ** 0.5
    results_dict["ridge_rmse"] = pred_df["Ridge_Difference"].mean() ** 0.5
    results_dict["elasticnet_rmse"] = pred_df["Elastic_Difference"].mean() ** 0.5
    results_dict["y_mean"] = y.mean()
    results_dict["y_stddev"] = y.std()
    created_at_utc = datetime.now(timezone.utc)
    created_at_utc = created_at_utc.strftime("%Y-%m-%d %H:%M:%S")
    results_dict["created_at_utc"] = str(created_at_utc)
    # Collect tuned hyperparameters
    param_summary = {
        "lasso_alpha": lasso_model.alpha_,
        "ridge_alpha": ridge_model.alpha_,
        "elasticnet_alpha": elastic_model.alpha_,
        "elasticnet_l1_ratio": elastic_model.l1_ratio_,
    }

    # Convert to JSON-safe string for storage
    results_dict["parameters_used"] = str(param_summary)

    # Overall completion timing
    total_duration = (datetime.now() - overall_start).total_seconds()
    # print(f"✅ All models complete! Total time: {total_duration:.1f}s")
    # print(f"   📊 Results: Lasso RMSE={results_dict['lasso_rmse']:.3f}, Ridge RMSE={results_dict['ridge_rmse']:.3f}, ElasticNet RMSE={results_dict['elasticnet_rmse']:.3f}")

    # Ridge performance analysis
    # if results_dict["ridge_rmse"] < results_dict["lasso_rmse"]:
    #     improvement = results_dict["lasso_rmse"] - results_dict["ridge_rmse"]
    #     print(f"   🏆 Ridge WINS by {improvement:.4f} RMSE points!")
    # else:
    #     deficit = results_dict["ridge_rmse"] - results_dict["lasso_rmse"]
    #     print(f"   😔 Ridge loses by {deficit:.4f} RMSE points")

    return results_dict


def upload_sim_resultset(snowflake_connection, results_dict):
    """
    Uploads the results of regularization simulations to a Snowflake database.

    This function takes a dictionary of simulation results and inserts them into
    a predefined table in the Snowflake database. The table stores various metrics
    and parameters related to the simulation, including RMSE values and derivation
    times for different regularization models.

    Args:
        snowflake_connection: A connection object to the Snowflake database.
        results_dict (dict): A dictionary containing the simulation results, including
                             metrics like RMSE and derivation times for each model.

    Returns:
        None
    """
    created_at_utc = datetime.now(timezone.utc)
    created_at_utc = created_at_utc.strftime("%Y-%m-%d %H:%M:%S")
    results_query = f"""
    INSERT INTO SANDBOX_DB.BENJAMINKNIGHT.REGULARIZATION_SIMULATION_RESULTS
    (NUM_FEATURES, NOBS, COVARIANCE_STDDEV, SPARSITY, SIGNAL_TO_NOISE, 
     VCOV_TOLERANCE, MAX_ITER, ALPHAS, L1_RATIOS, CV_FOLDS, TEST_RATIO, 
     LASSO_RMSE, RIDGE_RMSE, ELASTIC_NET_RMSE, LASSO_DERIVATION_SEC,
     RIDGE_DERIVATION_SEC, ELASTIC_NET_DERIVATION_SEC, Y_MEAN, Y_STDDEV, 
     CREATED_AT_UTC, PARAMETERS_USED, COVARIANCE_MEAN)
     VALUES ({results_dict['num_features']}, {results_dict['nobs']}, {results_dict['covariance_std']}, 
             {results_dict['sparsity']}, {results_dict['signal_to_noise']}, 
            '{str(results_dict['vcov_std_tol'])}', {results_dict['max_iterations']}, 
            '{str(results_dict['candidate_alpha_values'])}', '{str(results_dict['candidate_l1_ratios'])}', 
             {results_dict['cv_folds_used']}, {results_dict['test_ratio']}, {results_dict['lasso_rmse']}, 
             {results_dict['ridge_rmse']}, {results_dict['elasticnet_rmse']},  
             {results_dict['lasso_derivation_time']}, {results_dict['ridge_derivation_time']}, 
             {results_dict['elastc_derivation_time']}, {results_dict['y_mean']}, {results_dict['y_stddev']},
            '{created_at_utc}',  '{results_dict["parameters_used"]}', {results_dict['covariance_mean']})
    """
    snowflake_connection.cursor().execute(results_query)


def run_simulation(
    num_features,
    covariance_stddev,
    covariance_mean,
    sparsity,
    signal_to_noise,
    nobs,
    test_ratio,
    vcov_tolerance,
    max_iter,
    alphas,
    l1_ratios,
    cv_folds,
):
    """
    Run a simulation to generate correlated data, fit regularization models, and upload results.

    This function generates a correlation matrix, simulates regression coefficients,
    creates correlated data, and runs regularization simulations using Lasso, Ridge,
    ElasticNet, and LARS models. The results are then uploaded to a Snowflake database.

    Args:
        num_features (int): Number of features in the generated data.
        covariance_stddev (float): Target standard deviation for the covariance matrix.
        sparsity (float): Fraction of coefficients set to zero (0 = dense, 1 = fully sparse).
        signal_to_noise (float): Signal-to-noise ratio for the generated response variable.
        nobs (int): Number of observations to generate.
        test_ratio (float): Proportion of the dataset to include in the test split.
        vcov_tolerance (float): Tolerance for the variance-covariance standard deviation.
        max_iter (int): Maximum number of iterations for model fitting.
        alphas (list): List of alpha values to consider for Lasso and ElasticNet.
        l1_ratios (list): List of L1 ratios to consider for ElasticNet.
        cv_folds (int): Number of cross-validation folds.

    Returns:
        dict: A dictionary containing the RMSE and derivation time for each model,
              along with other simulation parameters.
    """
    # print(f"\n🚀 SIMULATION START: {num_features} features, {nobs:,} obs, stddev={covariance_stddev:.3f}, mean={covariance_mean:.3f}")
    overall_start = datetime.now()

    vcov = calibrate_corr_matrix(
        n=num_features,
        target_std=covariance_stddev,
        target_mean=covariance_mean,
        tol=vcov_tolerance,
        max_iter=max_iter,
        initial_guess=None,
    )
    betas = simulate_correlated_coefficients(Sigma=vcov[0], sparsity=sparsity)
    data = gen_correlated_data(corr_matrix=vcov[0], nobs=nobs, mean=None)
    data["Y"] = generate_response(
        X=data, beta=betas, signal_to_noise_ratio=signal_to_noise
    )
    results = run_regularization_sims(
        data=data,
        covariance_std=covariance_stddev,
        covariance_mean=covariance_mean,
        sparsity=sparsity,
        test_ratio=test_ratio,
        snr=signal_to_noise,
        vcov_std_tol=vcov_tolerance,
        max_iters=max_iter,
        allowed_alphas=alphas,
        allowed_l1_ratios=l1_ratios,
        cv_folds=cv_folds,
    )

    # Explicitly call garbage collection to free up memory
    gc.collect()

    # Final timing
    total_duration = (datetime.now() - overall_start).total_seconds()
    # print(f"🎉 SIMULATION COMPLETE in {total_duration:.1f}s")
    # print("-" * 60)

    return results


def save_results_to_csv(results_list, filename="results_temp.csv"):
    # Define the order of columns to match the Snowflake table
    fieldnames = [
        "num_features",
        "nobs",
        "covariance_std",
        "sparsity",
        "signal_to_noise",
        "vcov_std_tol",
        "max_iterations",
        "candidate_alpha_values",
        "candidate_l1_ratios",
        "cv_folds_used",
        "test_ratio",
        "lasso_rmse",
        "ridge_rmse",
        "elasticnet_rmse",
        "lasso_derivation_time",
        "ridge_derivation_time",
        "elastc_derivation_time",
        "y_mean",
        "y_stddev",
        "created_at_utc",
        "parameters_used",
        "covariance_mean",
    ]

    # Check if file exists to determine mode and whether to write header
    file_exists = os.path.isfile(filename)

    # Open the CSV file in append mode if it exists, write mode if it doesn't
    mode = "a" if file_exists else "w"

    with open(filename, mode=mode, newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # Write header only if creating a new file
        if not file_exists:
            writer.writeheader()
        # Write each result dictionary to the CSV
        for result in results_list:
            writer.writerow(result)


def write_results_to_snowflake(result_csv):
    # Define the column names to match the Snowflake table (in order matching CSV)
    column_names = [
        "NUM_FEATURES",
        "NOBS",
        "COVARIANCE_STDDEV",
        "SPARSITY",
        "SIGNAL_TO_NOISE",
        "VCOV_TOLERANCE",
        "MAX_ITER",
        "ALPHAS",
        "L1_RATIOS",
        "CV_FOLDS",
        "TEST_RATIO",
        "LASSO_RMSE",
        "RIDGE_RMSE",
        "ELASTIC_NET_RMSE",
        "LASSO_DERIVATION_SEC",
        "RIDGE_DERIVATION_SEC",
        "ELASTIC_NET_DERIVATION_SEC",
        "Y_MEAN",
        "Y_STDDEV",
        "CREATED_AT_UTC",
        "PARAMETERS_USED",  # This was missing!
        "COVARIANCE_MEAN",
    ]

    # Read the CSV file without a header and specify column names
    df = pd.read_csv(result_csv, header=0, names=column_names)

    # Ensure all column names are strings
    df.columns = df.columns.astype(str)

    # Establish a connection to Snowflake
    snowflake_connection = snowflake.connector.connect(
        user="BENJAMINKNIGHT",
        account="INSTACART-INSTACART",
        authenticator="externalbrowser",
        role="INSTACART_DEVELOPER_ROLE",
    )
    # cursor = snowflake_connection.cursor()
    # cursor.execute(f'USE DATABASE SNADBOX_DB;')
    # cursor.execute(f'USE SCHEMA BENJAMINKNIGHT;')
    # Use write_pandas to upload the DataFrame to Snowflake
    success, nchunks, nrows, _ = write_pandas(
        conn=snowflake_connection,
        df=df,
        database="SANDBOX_DB",
        schema="BENJAMINKNIGHT",
        table_name="REGULARIZATION_SIMULATION_RESULTS",
    )
    return success, nchunks, nrows, _
