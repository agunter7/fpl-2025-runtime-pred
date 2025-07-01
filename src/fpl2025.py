"""
To replicate results in the FPL2025 paper.
Open-Source FPGA Routing Runtime Prediction for Improved Productivity via Smart Route Termination.
"""
import os
import random
from typing import Optional
from constants import LabelIdx, FLAGSHIP_CIRCUITS, TINY_CIRCUITS, AGILEX_CIRCUITS, STRATIXIV_CIRCUITS, \
    STRATIX10_CIRCUITS
from time import time
from data_generator import get_dataset
import numpy as np
from os import path
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import r2_score, accuracy_score, matthews_corrcoef, balanced_accuracy_score, roc_auc_score
from sklearn.calibration import calibration_curve
from predictors import CustomClassifier, CustomRegressor, SimulatorPredictorFPL, SimulatorPredictorFCCM
import pickle
from simulator import sim_routing_attempt_fpl


def perform_routing_sim_evaluations_fpl2025(
        target_predictor: str,
        naivety_constant: Optional[float] = None,
        min_conf_to_extd_limit: Optional[float] = None,
        randomize_circuit_order: bool = True,
        route_time_budget: float = 30e6,  # 30 million seconds, effectively no limit. The entire dataset is ~25 mil.
        filter_time_limit: Optional[int] = None
):
    """
    Simulate FPGA routing with early exit.
    :param target_predictor: The predictor to simulate as driving early exit.
    :param naivety_constant: If a routing iteration takes less time than this, don't make a prediction for early exit.
    :param min_conf_to_extd_limit: Minimum prediction confidence to extend user-provided routing attempt time limit.
    :param randomize_circuit_order: Randomize the order of experimentation?
    :param route_time_budget: Experiment-wide routing time budget, amount of time available across all attempts.
    :param filter_time_limit: User-provided routing attempt time limit (applied per routing run)
    :return:
    """
    if target_predictor == "ml-iter":
        target_label_reg = LabelIdx.ROUTE_ITER
    else:
        target_label_reg = LabelIdx.NODE_TRAV

    base_path = path.dirname(__file__)

    # Get circuit data to simulate
    fccm_circuit_data_dir_path = path.abspath(path.join(
        # The FCCM MoE uses different features from the other ML models, but the classifiers and regressors
        #   in the FCCM MoE use the same features as each other.
        base_path, "..", "routing_sim_data", "class-reg_tuned_features_norm_fccm"
    ))
    class_circuit_data_dir_path = path.abspath(path.join(
        base_path, "..", "routing_sim_data", "class_tuned_features_norm"
    ))
    if target_label_reg == LabelIdx.NODE_TRAV:
        reg_circuit_data_dir_path = path.abspath(path.join(
            base_path, "..", "routing_sim_data", "reg_tuned_features_abs"
        ))
    elif target_label_reg == LabelIdx.ROUTE_ITER:
        # This uses physically normalized features, so just use classification features
        reg_circuit_data_dir_path = path.abspath(path.join(
            base_path, "..", "routing_sim_data", "class_tuned_features_norm"
        ))
    else:
        reg_circuit_data_dir_path = None
        exit(f"Bad target_label_reg for selecting sim data {target_label_reg}")
    # First verify that the classification and regression data match
    for X_filename in os.listdir(class_circuit_data_dir_path):
        reg_circuit_data_path = path.join(
            reg_circuit_data_dir_path, X_filename
        )
        fccm_circuit_data_path = path.join(
            fccm_circuit_data_dir_path, X_filename
        )
        assert path.exists(reg_circuit_data_path)
        assert path.exists(fccm_circuit_data_path)

    # Determine testing parameters (window range)
    if target_predictor == "ml-exp-val":
        experiment_ranges = {
            1800: (20,),
            3600: (15,),
            5400: (5,),
            7200: (5,),
        }
    elif target_predictor == "ml-iter":
        experiment_ranges = {
            # window range = 5 was found to be best in all cases
            1800: (5,),
            3600: (5,),
            5400: (5,),
            7200: (5,),
        }
    elif target_predictor == "naive":
        experiment_ranges = {
            1800: (1,),
            3600: (1,),
            5400: (1,),
            7200: (1,),
        }
    else:
        exit(f"Bad target predictor name '{target_predictor}'")
        experiment_ranges = {}

    # For each testing parameter, run simulation flow on all stratix circuits
    full_results = {}
    full_results_by_circuit = {}
    for time_limit in experiment_ranges.keys():
        if time_limit != filter_time_limit:
            continue
        if time_limit not in full_results.keys():
            full_results[time_limit] = {}
            full_results_by_circuit[time_limit] = {}
        window_range = experiment_ranges[time_limit]
        for window_size in window_range:
            print(f"Running experiment\n"
                  f"target predictor = {target_predictor}\n"
                  f"time limit = {time_limit}\n"
                  f"window size = {window_size}\n" +
                  f"naivety const. = {naivety_constant}\n"
                  f"min conf. to extend time limit = {min_conf_to_extd_limit}\n"
                  f"random circuit order = {randomize_circuit_order}\n"
                  f"route time budget = {route_time_budget}\n"
                  f"filter time limit = {filter_time_limit}")
            experiment_results = {
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
                "true_iters_routed": 0,
                "true_time_routed": 0,
                "true_time_ml": 0,
                "true_iters_wasted": 0,
                "true_time_routed_wasted": 0,
                "true_time_ml_wasted": 0,
                "experiment_time": None,
            }
            experiment_results_by_circuit = {}
            experiment_start = time()
            list_of_class_circuit_files = os.listdir(class_circuit_data_dir_path)
            if randomize_circuit_order:
                random.seed(0)  # Place this here so the shuffled order is constant across loop iterations
                random.shuffle(list_of_class_circuit_files)
            for X_filename in list_of_class_circuit_files:
                if not X_filename.endswith("X.csv"):
                    # This will prevent us from repeating simulation of the same circuit data
                    continue
                y_filename = X_filename.replace("X.csv", "y.csv")
                time_filename = X_filename.replace("X.csv", "time.csv")
                totals_filename = X_filename.replace("X.csv", "totals.csv")
                if target_predictor == "ml-iter":
                    # Uses FCCM MoE architecture which has different features from the rest of the models
                    class_X_path = path.join(
                        fccm_circuit_data_dir_path, X_filename
                    )
                    reg_X_path = class_X_path
                else:
                    class_X_path = path.join(
                        class_circuit_data_dir_path, X_filename
                    )
                    reg_X_path = path.join(
                        reg_circuit_data_dir_path, X_filename
                    )
                y_path = path.join(
                    class_circuit_data_dir_path, y_filename
                )
                time_path = path.join(
                    class_circuit_data_dir_path, time_filename
                )
                totals_path = path.join(
                    class_circuit_data_dir_path, totals_filename
                )
                X_class = np.genfromtxt(class_X_path, delimiter=',', dtype=float, skip_header=1)
                X_reg = np.genfromtxt(reg_X_path, delimiter=',', dtype=float, skip_header=1)
                y = np.genfromtxt(y_path, delimiter=',', skip_header=1)
                assert X_class.shape[0] == X_reg.shape[0] == y.shape[0]
                timing = np.genfromtxt(time_path, delimiter=',', dtype=float, skip_header=1)
                totals = np.genfromtxt(totals_path, delimiter=',', skip_header=1)
                circuit_total_route_time = totals[LabelIdx.TOT_PURE_ROUTE_TIME]
                if 0 < circuit_total_route_time < time_limit/2:  # Unrouted circuits have negative labels, so 0<time
                    # To try to maintain reasonable parity between circuits in experiments,
                    #   don't try routing circuits which complete in less than half the user time limit.
                    # This means the greatest ratio between circuit runtimes will be 4:1,
                    #   as the shortest is half the time limit and longest is double the time limit extended route time)
                    continue

                circuit_start = time()
                circuit_name = X_filename.split('_')[1]
                if circuit_name not in experiment_results_by_circuit.keys():
                    experiment_results_by_circuit[circuit_name] = {
                        "tn": 0,
                        "fp": 0,
                        "fn": 0,
                        "tp": 0,
                        "true_iters_routed": 0,
                        "true_time_routed": 0,
                        "true_time_ml": 0,
                        "true_iters_wasted": 0,
                        "true_time_routed_wasted": 0,
                        "true_time_ml_wasted": 0,
                        "experiment_time": 0,
                    }
                circuit_results = experiment_results_by_circuit[circuit_name]

                # Create simulator predictor for this circuit
                if target_predictor == "ml-exp-val":
                    model_dir_path = path.abspath(path.join(
                        base_path, "..", "trained_models", "fpl2025_moe_exp_val", circuit_name
                    ))
                    classifier_path = path.join(
                        model_dir_path,
                        "c-iter-1000_hgb_fg1100_ftune0_htune0_fpl-excl-timeout-custom-estimator.pkl"
                    )
                    quantiles = (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)
                    quantile_models = {}
                    for quantile in quantiles:
                        quantile_path = path.join(
                            model_dir_path,
                            f"{quantile}q-trav-15e10_hgb_fg1100_ftune0_htune0_fpl-custom-estimator.pkl"
                        )
                        with open(quantile_path, 'rb') as regressor_infile:
                            quantile_models[quantile] = pickle.load(regressor_infile)
                    regressor_path = path.join(
                        model_dir_path,
                        f"r-trav-15e10_hgb_fg1100_ftune0_htune0_fpl-custom-estimator.pkl"
                    )
                    with open(regressor_path, 'rb') as regressor_infile:
                        mean_regressor = pickle.load(regressor_infile)
                    with open(classifier_path, 'rb') as classifier_infile:
                        classifier = pickle.load(classifier_infile)
                    # Create predictor to use in simulation
                    sim_sys = SimulatorPredictorFPL(
                        name="exp-val-sys", classifier=classifier, target_label_regression=target_label_reg,
                        use_overuse_features=True, use_switchbox_features=True,
                        use_wlpa_features=False, use_ncpr_features=False,
                        q5=quantile_models[5], q10=quantile_models[10], q20=quantile_models[20],
                        q30=quantile_models[30],
                        q40=quantile_models[40], q50=quantile_models[50], q60=quantile_models[60],
                        q70=quantile_models[70],
                        q80=quantile_models[80], q90=quantile_models[90], q95=quantile_models[95],
                        mean_regressor=mean_regressor
                    )
                elif target_predictor == "ml-iter":
                    model_dir_path = path.abspath(path.join(
                        base_path, "..", "trained_models", "fpl2025_fccm_moe", circuit_name
                    ))
                    classifiers = {}
                    regressors = {}
                    for interval in (150, 250, 400, 1000):
                        classifier_path = path.join(
                            model_dir_path, f"f{interval}-iter_hgb_fg1100_ftune0_htune0_fccm.pkl"
                        )
                        regressor_path = path.join(
                            model_dir_path, f"r{interval}-iter_hgb_fg1100_ftune0_htune0_fccm.pkl"
                        )
                        with open(classifier_path, 'rb') as classifier_infile:
                            classifier = CustomClassifier(
                                name=f"f{interval}",
                                estimator=pickle.load(classifier_infile)
                            )
                        with open(regressor_path, 'rb') as regressor_infile:
                            regressor = CustomRegressor(
                                name=f"r{interval}",
                                estimator=pickle.load(regressor_infile)
                            )
                        classifiers[interval] = classifier
                        regressors[interval] = regressor
                    # Create predictor to use in simulation
                    sim_sys = SimulatorPredictorFCCM(
                        f150=classifiers[150], f250=classifiers[250], f400=classifiers[400], f1000=classifiers[1000],
                        r150=regressors[150], r250=regressors[250], r400=regressors[400], r1000=regressors[1000],
                        use_overuse_features=True, use_switchbox_features=True,
                        use_wlpa_features=True, use_ncpr_features=True, name="fccm-moe",

                    )
                elif target_predictor == "naive":
                    sim_sys = None
                else:
                    sim_sys = None
                    exit(f"Unsupported choice of prediction system")

                # Determine if early exit should be enabled
                if target_predictor == "naive":
                    enable_ee = False
                else:
                    enable_ee = True
                if target_predictor == "ml-iter":
                    # The early exit stopping criterion is now an iteration limit instead of a time limit
                    iter_ee_mode = True
                    is_naive_for_fast_routing = False
                else:
                    iter_ee_mode = False
                    is_naive_for_fast_routing = True
                # Determine if using expected value early exit mode
                if target_predictor == "ml-exp-val":
                    exp_val_mode = True
                else:
                    exp_val_mode = False

                results = sim_routing_attempt_fpl(
                    X_class=X_class, X_reg=X_reg, y=y, timing=timing, init_time_limit=time_limit,
                    sim_predictor=sim_sys, window_size=window_size,
                    totals=totals, enable_ee=enable_ee, iter_ee_mode=iter_ee_mode,
                    is_naive_for_fast_routing=is_naive_for_fast_routing,
                    naivety_constant=naivety_constant,
                    exp_val_mode=exp_val_mode, min_conf_to_extd_limit=min_conf_to_extd_limit
                )
                if results["is_routable"]:
                    if results["routed_all_iters"]:
                        experiment_results["tp"] += 1
                        circuit_results["tp"] += 1
                    else:
                        experiment_results["fn"] += 1
                        experiment_results["true_iters_wasted"] += results["iters_routed"] - 1
                        experiment_results["true_time_routed_wasted"] += results["true_time_routed"]
                        experiment_results["true_time_ml_wasted"] += results["true_time_ml"]
                        circuit_results["fn"] += 1
                        circuit_results["true_iters_wasted"] += results["iters_routed"] - 1
                        circuit_results["true_time_routed_wasted"] += results["true_time_routed"]
                        circuit_results["true_time_ml_wasted"] += results["true_time_ml"]
                else:
                    experiment_results["true_iters_wasted"] += results["iters_routed"] - 1
                    experiment_results["true_time_routed_wasted"] += results["true_time_routed"]
                    experiment_results["true_time_ml_wasted"] += results["true_time_ml"]
                    circuit_results["true_iters_wasted"] += results["iters_routed"] - 1
                    circuit_results["true_time_routed_wasted"] += results["true_time_routed"]
                    circuit_results["true_time_ml_wasted"] += results["true_time_ml"]
                    if results["routed_all_iters"]:
                        experiment_results["fp"] += 1
                        circuit_results["fp"] += 1
                    else:
                        experiment_results["tn"] += 1
                        circuit_results["tn"] += 1
                experiment_results["true_iters_routed"] += results["iters_routed"] - 1
                experiment_results["true_time_routed"] += results["true_time_routed"]
                experiment_results["true_time_ml"] += results["true_time_ml"]
                circuit_results["true_iters_routed"] += results["iters_routed"] - 1
                circuit_results["true_time_routed"] += results["true_time_routed"]
                circuit_results["true_time_ml"] += results["true_time_ml"]
                circuit_end = time()
                circuit_time = circuit_end - circuit_start
                circuit_results["experiment_time"] += circuit_time
                experiment_true_time_routed = experiment_results["true_time_routed"]
                if experiment_true_time_routed >= route_time_budget:
                    print(f"Ending experiment,"
                          f" time routed ({int(experiment_true_time_routed)}) > budget {route_time_budget}")
                    break
            # Tabulate results
            experiment_end = time()
            experiment_time = experiment_end-experiment_start
            experiment_results["experiment_time"] = experiment_time
            print(f"Experiment took {experiment_time:.1f}s")
            tp = experiment_results["tp"]
            total_time = experiment_results["true_time_routed"] + experiment_results["true_time_ml"]
            print(f"ROUTING SUCCESS RATE: {tp} CIRCUITS ROUTED")
            print(f"TOTAL (SIMULATED) TIME: {total_time:.1f}s\n")
            full_results[time_limit][window_size] = experiment_results
            full_results_by_circuit[time_limit][window_size] = experiment_results_by_circuit


def test_stratix_circuits(
        is_class: bool = True,
        target_label_reg: LabelIdx = LabelIdx.NODE_TRAV,
        is_loocv: bool = False,
        random_state: int = 0,
        do_compute_platform_testing: bool = False
):
    """
    Test on each stratix circuit's data separately and aggregate results.
    :param is_class: Is this classification? Or regression?
    :param target_label_reg: Target regression label (i.e. node traversals vs. route iterations)
    :param is_loocv: Use leave one (circuit) out cross-validation?
    :param random_state:
    :param do_compute_platform_testing: Should this test for simulated results on a slower/faster compute platform?
    :return:
    """
    if do_compute_platform_testing:
        assert target_label_reg == LabelIdx.TOT_PURE_ROUTE_TIME
    # Experimental parameters
    flagship_circuit_list = list(FLAGSHIP_CIRCUITS)
    agilex_circuit_list = list(AGILEX_CIRCUITS)
    for tiny_circuit in TINY_CIRCUITS:
        try:
            flagship_circuit_list.remove(tiny_circuit)
        except Exception as e:
            print(e)
        try:
            agilex_circuit_list.remove(tiny_circuit)
        except Exception as e:
            print(e)
    train_arch_circuits = {
        "flagship": flagship_circuit_list,
        "agilex": agilex_circuit_list,
        "stratixiv": list(STRATIXIV_CIRCUITS),
        "stratix10": list(STRATIX10_CIRCUITS)
    }

    if is_class:
        target_label = LabelIdx.ROUTE_ITER
        loss_criterion = 'log_loss'
        max_iter = 45
        l2_regularization = 0
        learning_rate = 0.035
        class_weight = "balanced"
    else:
        target_label = target_label_reg
        loss_criterion = 'squared_error'
        max_iter = 60
        l2_regularization = 1.0
        learning_rate = 0.14
        class_weight = None
    validation_fraction = None
    early_stopping = False
    scoring = None
    warm_start = False

    base_path = path.dirname(__file__)

    if is_loocv:
        # Get training data
        X_full = None
        y_full = None
        circuit_index_range_map = {}
        for train_arch_name in train_arch_circuits.keys():
            train_circuit_list = train_arch_circuits[train_arch_name]
            for train_circuit_name_ugly in train_circuit_list:
                train_circuit_name_pretty = train_circuit_name_ugly.replace('_', '-')
                # Load dataset (circuit data)
                if is_class:
                    dataset_name = f"{train_arch_name}_{train_circuit_name_pretty}_class_norm"
                else:
                    dataset_name = f"{train_arch_name}_{train_circuit_name_pretty}_reg"
                X_circuit, y_circuit, feature_names, _ = get_dataset(
                    is_classification=is_class,
                    set_name=dataset_name,
                    is_sim=False,
                    target_label=target_label,
                    subdir="final_eval_by_circuit"
                )
                if len(y_circuit) > 0:
                    if X_full is None:
                        X_full = X_circuit
                        start_idx = 0
                        end_idx = X_full.shape[0]
                    else:
                        start_idx = X_full.shape[0]
                        X_full = np.concatenate(
                            (X_full, X_circuit)
                        )
                        end_idx = X_full.shape[0]
                    circuit_index_range_map[train_circuit_name_ugly] = (start_idx, end_idx)
                    if y_full is None:
                        y_full = y_circuit
                    else:
                        y_full = np.concatenate(
                            (y_full, y_circuit)
                        )
                else:
                    circuit_index_range_map[train_circuit_name_ugly] = None
        # Define model to be trained later
        if is_class:
            model = HistGradientBoostingClassifier(
                loss=loss_criterion, validation_fraction=validation_fraction, early_stopping=early_stopping,
                random_state=random_state, scoring=scoring, warm_start=warm_start,
                max_iter=max_iter, l2_regularization=l2_regularization, learning_rate=learning_rate,
                class_weight=class_weight
            )
        else:
            model = HistGradientBoostingRegressor(
                loss=loss_criterion, validation_fraction=validation_fraction, early_stopping=early_stopping,
                random_state=random_state, scoring=scoring, warm_start=warm_start,
                max_iter=max_iter, l2_regularization=l2_regularization, learning_rate=learning_rate
            )
    else:
        # Load model
        model_dir_path = path.abspath(path.join(
            base_path, "..", "trained_models", "final_eval"
        ))
        if is_class:
            model_path = path.join(
                model_dir_path, "c-iter-1000_hgb_fg1100_ftune0_htune0.pkl"
            )
            with open(model_path, 'rb') as classifier_infile:
                model = CustomClassifier(
                    name="hgb-class",
                    estimator=pickle.load(classifier_infile)
                )
        else:
            if target_label_reg == LabelIdx.NODE_TRAV:
                model_path = path.join(
                    model_dir_path, "r-trav-15e10_hgb_fg1100_ftune0_htune0.pkl"
                )
                with open(model_path, 'rb') as regressor_infile:
                    model = CustomRegressor(
                        name="hgb-reg",
                        estimator=pickle.load(regressor_infile)
                    )
            else:
                model: Optional[CustomRegressor, CustomClassifier] = None
                exit("Need to determine path for other models")
        circuit_index_range_map = None
        X_full = None
        y_full = None

    test_candidate_circuit_list = list(STRATIX10_CIRCUITS) + list(STRATIXIV_CIRCUITS)
    y_test_agg = None
    if not is_class and is_loocv:
        for test_circuit_name_ugly in test_candidate_circuit_list:
            for test_arch_name in ("stratix10", "stratixiv"):
                if test_arch_name not in test_circuit_name_ugly:
                    continue  # This circuit is for the other stratix architecture

                mask_test = np.zeros(X_full.shape[0], dtype=bool)

                start_idx_test, end_idx_test = circuit_index_range_map[test_circuit_name_ugly]
                assert start_idx_test is not None and end_idx_test is not None
                mask_test[start_idx_test:end_idx_test] = True
                y_test_circuit = y_full[mask_test]
                if y_test_agg is None:
                    y_test_agg = np.copy(y_test_circuit)
                else:
                    y_test_agg = np.concatenate(
                        (y_test_agg, y_test_circuit)
                    )

    # Test
    y_test_agg = None  # Reset in case we used it previously
    test_count = 0
    no_unroutable_count = 0  # Counting how many circuits have no false/unroutable data, this
    #   can happen if they are only ever unroutable due to timeout rather than route iteration
    y_pred_test_agg = None
    r2_test_avg = 0
    largest_train_label = None
    for test_circuit_name_ugly in test_candidate_circuit_list:
        for test_arch_name in ("stratix10", "stratixiv"):
            if test_arch_name not in test_circuit_name_ugly:
                continue  # This circuit is for the other stratix arch
            print(test_circuit_name_ugly)

            if is_loocv:
                mask_train = np.ones(X_full.shape[0], dtype=bool)
                mask_test = np.zeros(X_full.shape[0], dtype=bool)

                start_idx_test, end_idx_test = circuit_index_range_map[test_circuit_name_ugly]
                if start_idx_test is None:
                    continue
                mask_test[start_idx_test:end_idx_test] = True
                mask_train[start_idx_test:end_idx_test] = False
                # All stratix circuits exist for both Stratix10 and Stratixiv
                # We should exclude both from training
                if "stratixiv" in test_circuit_name_ugly:
                    excluded_circuit_name_ugly = test_circuit_name_ugly.replace("stratixiv", "stratix10")
                else:
                    excluded_circuit_name_ugly = test_circuit_name_ugly.replace("stratix10", "stratixiv")
                if excluded_circuit_name_ugly in STRATIX10_CIRCUITS or excluded_circuit_name_ugly in STRATIXIV_CIRCUITS:
                    # The circuit CHERI does not exist for stratix10, so the if clause accounts for that anomaly
                    start_idx_test2, end_idx_test2 = circuit_index_range_map[excluded_circuit_name_ugly]
                    assert start_idx_test2 is not None and end_idx_test2 is not None
                    mask_train[start_idx_test2:end_idx_test2] = False

                X_train, y_train = X_full[mask_train], y_full[mask_train]
                largest_train_label = np.max(y_train)

                X_test, y_test = X_full[mask_test], y_full[mask_test]

                model.fit(
                    X=X_train, y=y_train
                )
            else:
                test_circuit_name_pretty = test_circuit_name_ugly.replace('_', '-')
                if is_class:
                    dataset_name = f"{test_arch_name}_{test_circuit_name_pretty}_class_norm"
                else:
                    dataset_name = f"{test_arch_name}_{test_circuit_name_pretty}_reg"
                X_circuit, y_circuit, _, _ = get_dataset(
                    is_classification=is_class,
                    set_name=dataset_name,
                    is_sim=False,
                    target_label=target_label,
                    subdir="final_eval_by_circuit"
                )
                X_test, y_test = X_circuit, y_circuit

            num_labels_test = y_test.shape[0]
            if num_labels_test == 0:
                print("Skipping")
                continue
            if is_class:
                num_true_labels_test = sum(y_test)
                if num_labels_test == num_true_labels_test:
                    no_unroutable_count += 1

            # Test
            if not is_class:
                # Evaluate
                y_pred_test = np.clip(model.predict(X=X_test), a_min=1, a_max=largest_train_label)
                r2_test = r2_score(y_pred=y_pred_test, y_true=y_test)
                r2_test_avg += r2_test * num_labels_test
            else:
                # Evaluate
                y_pred_test = model.predict(X=X_test)
            test_count += 1
            # Record y values for this circuit
            # if not completed_y_test_agg:
            if y_test_agg is None:
                y_test_agg = np.copy(y_test)
            else:
                y_test_agg = np.concatenate(
                    (y_test_agg, y_test)
                )
            if y_pred_test_agg is None:
                y_pred_test_agg = np.copy(y_pred_test)
            else:
                y_pred_test_agg = np.concatenate(
                    (y_pred_test_agg, y_pred_test)
                )

    if is_class:
        acc_test_agg = accuracy_score(y_pred=y_pred_test_agg, y_true=y_test_agg)
        mcc_test_agg = matthews_corrcoef(y_pred=y_pred_test_agg, y_true=y_test_agg)
        print("OVERALL")
        print(f"ACCURACY: {acc_test_agg:.3f}, MCC: {mcc_test_agg:.3f}")
    else:
        if do_compute_platform_testing:
            for relative_test_speed in (
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                    1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2
            ):
                print(f"relative test speed = {relative_test_speed}")
                new_y_test_agg = np.copy(y_test_agg) / relative_test_speed
                r2_test_agg = r2_score(y_pred=y_pred_test_agg, y_true=new_y_test_agg)
                r2_test_avg /= y_test_agg.shape[0]
                print(f"R2: {r2_test_agg: .3f}")
        else:
            r2_test_agg = r2_score(y_pred=y_pred_test_agg, y_true=y_test_agg)
            r2_test_avg /= y_test_agg.shape[0]
            print("OVERALL")
            print(f"R2: {r2_test_agg: .3f}")


def analyze_exp_val_moe_classifiers(
        exclude_timeout: bool
):
    """
    Inspect the performance of classifiers used in the Mixture of Experts system for node traversal prediction.
    :param exclude_timeout: Exclude timed-out runs from data? Else use model where timeouts are labelled as routable
    :return:
    """
    base_path = path.dirname(__file__)
    if exclude_timeout:
        subdir = "final_eval_by_circuit_exclude_timeout"
    else:
        subdir = "final_eval_by_circuit"

    base_model_dir = path.abspath(path.join(
        base_path, "..", "trained_models", "fpl2025_moe_exp_val"
    ))

    # Test
    y_test_agg = None
    test_count = 0
    no_unroutable_count = 0  # Counting how many circuits have no false/unroutable data, this
    #   can happen if they are only ever unroutable due to timeout rather than route iteration
    y_pred_test_agg = None
    y_pred_proba_test_agg = None
    test_candidate_circuit_list = list(STRATIX10_CIRCUITS) + list(STRATIXIV_CIRCUITS)
    for test_circuit_name_ugly in test_candidate_circuit_list:
        for test_arch_name in ("stratix10", "stratixiv"):
            if test_arch_name not in test_circuit_name_ugly:
                continue  # This circuit is for the other stratix arch
            print(test_circuit_name_ugly)
            test_circuit_name_pretty = test_circuit_name_ugly.replace('_', '-')

            dataset_name = f"{test_arch_name}_{test_circuit_name_pretty}_class_norm"
            X_test, y_test, feature_names, _ = get_dataset(
                is_classification=True,
                set_name=dataset_name,
                is_sim=False,
                target_label=LabelIdx.ROUTE_ITER,
                subdir=subdir
            )

            # Load classifier
            model_circuit_dir = path.join(
                base_model_dir, test_circuit_name_pretty
            )
            model_name = "c-iter-1000_hgb_fg1100_ftune0_htune0_fpl-"
            if exclude_timeout:
                model_name += "excl-timeout-custom-estimator.pkl"
            else:
                model_name += "routable-timeout-custom-estimator.pkl"

            model_path = path.join(
                model_circuit_dir, model_name
            )
            with open(model_path, 'rb') as infile:
                model: CustomClassifier = pickle.load(infile)

            num_labels_test = y_test.shape[0]
            assert num_labels_test != 0
            num_true_labels_test = sum(y_test)
            if num_true_labels_test == 0:
                no_unroutable_count += 1

            # Test
            # Evaluate
            y_pred_test = model.predict(X=X_test)
            assert num_true_labels_test != 0

            # Track data for analytics
            if y_pred_test_agg is None:
                y_pred_test_agg = y_pred_test
            else:
                y_pred_test_agg = np.concatenate(
                    (y_pred_test_agg, y_pred_test)
                )
            if y_test_agg is None:
                y_test_agg = y_test
            else:
                y_test_agg = np.concatenate(
                    (y_test_agg, y_test)
                )
            y_pred_proba_test = model.predict_proba(X=X_test)
            if y_pred_proba_test_agg is None:
                y_pred_proba_test_agg = y_pred_proba_test
            else:
                y_pred_proba_test_agg = np.concatenate(
                    (y_pred_proba_test_agg, y_pred_proba_test)
                )

            test_count += 1

    # Aggregate metrics
    bal_acc_test_agg = balanced_accuracy_score(y_pred=y_pred_test_agg, y_true=y_test_agg)
    roc_auc = roc_auc_score(y_score=y_pred_proba_test_agg[:, 1], y_true=y_test_agg)
    print("OVERALL")
    print(f"ROC-AUC, BAL. ACC.")
    print(f"{roc_auc: .3f}, {bal_acc_test_agg: .3f}")

    # Gauge calibration of model
    prob_true, prob_pred = calibration_curve(
        y_prob=y_pred_proba_test_agg[:, 1], y_true=y_test_agg,
        n_bins=20, strategy='uniform'
    )
    print("ACTUAL ACCURACY")
    print(prob_true)
    print("NOMINAL CONFIDENCE")
    print(prob_pred)


def analyze_fccm_moe_classifiers(
        interval: int
):
    """
    Inspect the performance of classifiers used in the Mixture of Experts system from FCCM2023 paper,
    A machine learning approach for predicting the difficulty of fpga routing problems.
    :param interval: Valid values: 150, 250, 400, 1000. To test different classifiers in the system.
    :return:
    """
    base_path = path.dirname(__file__)
    subdir = f"fccm_by_circuit_{interval}_exclude_timeout"

    base_model_dir = path.abspath(path.join(
        base_path, "..", "trained_models", "fpl2025_fccm_moe"
    ))

    # Test
    y_test_agg = None  # Reset in case we used it previously
    test_count = 0
    y_pred_test_agg = None
    y_pred_proba_test_agg = None
    test_candidate_circuit_list = list(STRATIX10_CIRCUITS) + list(STRATIXIV_CIRCUITS)
    for test_circuit_name_ugly in test_candidate_circuit_list:
        for test_arch_name in ("stratix10", "stratixiv"):
            if test_arch_name not in test_circuit_name_ugly:
                continue  # This circuit is for the other stratix arch
            print(test_circuit_name_ugly)
            test_circuit_name_pretty = test_circuit_name_ugly.replace('_', '-')

            dataset_name = f"{test_arch_name}_{test_circuit_name_pretty}_class_norm"
            X_test, y_test, feature_names, _ = get_dataset(
                is_classification=True,
                set_name=dataset_name,
                is_sim=False,
                target_label=LabelIdx.ROUTE_ITER,
                subdir=subdir
            )

            # Load classifier
            model_circuit_dir = path.join(
                base_model_dir, test_circuit_name_pretty
            )
            model_name = f"f{interval}-iter_hgb_fg1100_ftune0_htune0_fccm.pkl"
            model_path = path.join(
                model_circuit_dir, model_name
            )
            with open(model_path, 'rb') as infile:
                model = pickle.load(infile)

            num_labels_test = y_test.shape[0]
            assert num_labels_test != 0

            # Evaluate
            y_pred_test = model.predict(X=X_test)

            # Track data for analytics
            if y_pred_test_agg is None:
                y_pred_test_agg = y_pred_test
            else:
                y_pred_test_agg = np.concatenate(
                    (y_pred_test_agg, y_pred_test)
                )
            if y_test_agg is None:
                y_test_agg = y_test
            else:
                y_test_agg = np.concatenate(
                    (y_test_agg, y_test)
                )
            y_pred_proba_test = model.predict_proba(X=X_test)
            if y_pred_proba_test_agg is None:
                y_pred_proba_test_agg = y_pred_proba_test
            else:
                y_pred_proba_test_agg = np.concatenate(
                    (y_pred_proba_test_agg, y_pred_proba_test)
                )

            test_count += 1

    # Aggregate metrics
    bal_acc_test_agg = balanced_accuracy_score(y_pred=y_pred_test_agg, y_true=y_test_agg)
    roc_auc = roc_auc_score(y_score=y_pred_proba_test_agg[:, 1], y_true=y_test_agg)
    print("OVERALL")
    print(f"ROC-AUC, BAL. ACC.")
    print(f"{roc_auc: .3f}, {bal_acc_test_agg: .3f}")


def analyze_fccm_moe_regressors(
        interval_param: int
):
    """
    Inspect the performance of regressors used in the Mixture of Experts system from FCCM2023 paper,
    A machine learning approach for predicting the difficulty of fpga routing problems.
    :param interval_param: Valid values: 150, 250, 400, 1000. To test different regressors in the system.
    :return:
    """
    base_path = path.dirname(__file__)
    if interval_param is not None:
        subdir = f"fccm_by_circuit_{interval_param}"
    else:
        subdir = f"fccm_by_circuit_1000"

    base_model_dir = path.abspath(path.join(
        base_path, "..", "trained_models", "fpl2025_fccm_moe"
    ))

    # Test
    y_test_agg = None  # Reset in case we used it previously
    test_count = 0
    y_pred_test_agg = None
    test_candidate_circuit_list = list(STRATIX10_CIRCUITS) + list(STRATIXIV_CIRCUITS)
    sim_sys = None
    for test_circuit_name_ugly in test_candidate_circuit_list:
        for test_arch_name in ("stratix10", "stratixiv"):
            if test_arch_name not in test_circuit_name_ugly:
                continue  # This circuit is for the other stratix arch
            print(test_circuit_name_ugly)
            test_circuit_name_pretty = test_circuit_name_ugly.replace('_', '-')

            dataset_name = f"{test_arch_name}_{test_circuit_name_pretty}_reg_norm"
            X_test, y_test, feature_names, _ = get_dataset(
                is_classification=False,
                set_name=dataset_name,
                is_sim=False,
                target_label=LabelIdx.ROUTE_ITER,
                subdir=subdir
            )

            # Load classifier
            if interval_param is not None:
                model_circuit_dir = path.join(
                    base_model_dir, test_circuit_name_pretty
                )
                model_name = f"r{interval_param}-iter_hgb_fg1100_ftune0_htune0_fccm.pkl"

                model_path = path.join(
                    model_circuit_dir, model_name
                )
                with open(model_path, 'rb') as infile:
                    model = pickle.load(infile)
            else:
                model_dir_path = path.abspath(path.join(
                    base_path, "..", "trained_models", "fpl2025_fccm_moe", test_circuit_name_pretty
                ))
                classifiers = {}
                regressors = {}
                for interval in (150, 250, 400, 1000):
                    classifier_path = path.join(
                        model_dir_path, f"f{interval}-iter_hgb_fg1100_ftune0_htune0_fccm.pkl"
                    )
                    regressor_path = path.join(
                        model_dir_path, f"r{interval}-iter_hgb_fg1100_ftune0_htune0_fccm.pkl"
                    )
                    with open(classifier_path, 'rb') as classifier_infile:
                        classifier = CustomClassifier(
                            name=f"f{interval}",
                            estimator=pickle.load(classifier_infile)
                        )
                    with open(regressor_path, 'rb') as regressor_infile:
                        regressor = CustomRegressor(
                            name=f"r{interval}",
                            estimator=pickle.load(regressor_infile)
                        )
                    classifiers[interval] = classifier
                    regressors[interval] = regressor
                # Create predictor to use in simulation
                sim_sys = SimulatorPredictorFCCM(
                    f150=classifiers[150], f250=classifiers[250], f400=classifiers[400], f1000=classifiers[1000],
                    r150=regressors[150], r250=regressors[250], r400=regressors[400], r1000=regressors[1000],
                    use_overuse_features=False, use_switchbox_features=False,
                    use_wlpa_features=False, use_ncpr_features=False, name="fccm-moe",
                    explicitly_override_extraction_time=True

                )

            num_labels_test = len(y_test)
            if num_labels_test == 0:
                continue

            # Evaluate
            if interval_param is not None:
                y_pred_test = model.predict(X=X_test)
            else:
                y_pred_test = []
                for X_test_row in X_test:
                    X_test_row = X_test_row.reshape(1, -1)
                    _, iter_remaining_pred, _, _ = sim_sys.sim_predict(
                        X_row_class=X_test_row, X_row_reg=X_test_row, timing_row=None,
                        time_remaining_to_limit=None, iter_route_speed=None, num_iters_routed=None,
                        time_routed=None
                    )
                    if iter_remaining_pred is not None:
                        y_pred_test.append(iter_remaining_pred)
                    else:
                        y_pred_test.append(
                            sim_sys.r1000.estimator.predict(X_test_row)[0]
                        )
                y_pred_test = np.asarray(y_pred_test)

            # Track data for analytics
            if y_pred_test_agg is None:
                y_pred_test_agg = y_pred_test
            else:
                y_pred_test_agg = np.concatenate(
                    (y_pred_test_agg, y_pred_test)
                )
            if y_test_agg is None:
                y_test_agg = y_test
            else:
                y_test_agg = np.concatenate(
                    (y_test_agg, y_test)
                )

            test_count += 1

    r2_test_agg = r2_score(y_pred=y_pred_test_agg, y_true=y_test_agg)
    print("OVERALL R2")
    print(f"{r2_test_agg: .3f}")


def analyze_exp_val_moe_regressors(
        quantile: Optional[int] = None
):
    """
    Inspect the calibration of the quantile regressors trained for the FPL2025 paper's mixture of experts system.
    :param quantile: Determines the model to inspect. Valid values: 5, 10, 20 30, 40, 50, 60, 70, 80, 90, 95.
    :return:
    """
    base_path = path.dirname(__file__)
    subdir = "final_eval_by_circuit"

    base_model_dir = path.abspath(path.join(
        base_path, "..", "trained_models", "fpl2025_moe_exp_val"
    ))

    # Test
    y_test_agg = None  # Reset in case we used it previously
    test_count = 0
    y_pred_test_agg = None
    test_candidate_circuit_list = list(STRATIX10_CIRCUITS) + list(STRATIXIV_CIRCUITS)
    for test_circuit_name_ugly in test_candidate_circuit_list:
        for test_arch_name in ("stratix10", "stratixiv"):
            if test_arch_name not in test_circuit_name_ugly:
                continue  # This circuit is for the other stratix arch
            test_circuit_name_pretty = test_circuit_name_ugly.replace('_', '-')

            dataset_name = f"{test_arch_name}_{test_circuit_name_pretty}_reg"
            X_test, y_test, feature_names, _ = get_dataset(
                is_classification=False,
                set_name=dataset_name,
                is_sim=False,
                target_label=LabelIdx.NODE_TRAV,
                subdir=subdir
            )

            # Load classifier
            model_circuit_dir = path.join(
                base_model_dir, test_circuit_name_pretty
            )
            if quantile is not None:
                model_name = f"{quantile}q-trav-15e10_hgb_fg1100_ftune0_htune0_fpl-custom-estimator.pkl"
            else:
                model_name = f"r-trav-15e10_hgb_fg1100_ftune0_htune0_fpl-custom-estimator.pkl"

            model_path = path.join(
                model_circuit_dir, model_name
            )
            with open(model_path, 'rb') as infile:
                model: CustomRegressor = pickle.load(infile)

            num_labels_test = y_test.shape[0]
            if num_labels_test == 0:  # Can happen when filtering out test samples
                continue

            # Evaluate
            y_pred_test = model.predict(X=X_test)

            # Track data for analytics
            if y_pred_test_agg is None:
                y_pred_test_agg = y_pred_test
            else:
                y_pred_test_agg = np.concatenate(
                    (y_pred_test_agg, y_pred_test)
                )
            if y_test_agg is None:
                y_test_agg = y_test
            else:
                y_test_agg = np.concatenate(
                    (y_test_agg, y_test)
                )

            test_count += 1

    # Check calibration and quality of model
    if quantile is not None:
        num_samples = y_test_agg.shape[0]
        num_samples_below_quantile = np.sum(y_test_agg < y_pred_test_agg)
        print(f"NOMINAL QUANTILE = {quantile} ||| "
              f"ACTUAL ACHIEVED QUANTILE: {(num_samples_below_quantile / num_samples): .3f}")
    else:
        r2_test_agg = r2_score(y_pred=y_pred_test_agg, y_true=y_test_agg)
        print(f"OVERALL R2")
        print(f"{r2_test_agg: .3f}")


##########################################################################################
def reproduce_table_2_results(random_state):
    test_stratix_circuits(  # Table 2, direct time result
        is_class=False,
        target_label_reg=LabelIdx.TOT_PURE_ROUTE_TIME,
        is_loocv=True, random_state=random_state,
        do_compute_platform_testing=False
    )
    analyze_exp_val_moe_classifiers(
        # Table 2, Our classifier bal acc & roc-auc
        exclude_timeout=True
    )
    analyze_fccm_moe_classifiers(  # Table 2, prior classifier bal acc & roc-auc
        interval=1000
    )
    analyze_fccm_moe_regressors(  # Table 2, prior regressor R^2
        interval_param=1000
    )
    analyze_exp_val_moe_regressors(  # Table 2, Our regressor R^2
    )


def reproduce_figure_5_results(random_state):
    test_stratix_circuits(  # Figure 5 raw data for direct time case
        is_class=False,
        target_label_reg=LabelIdx.TOT_PURE_ROUTE_TIME,
        is_loocv=True, random_state=random_state,
        do_compute_platform_testing=True
    )


def reproduce_figure_6_results():
    for quantile in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95):  # Figure 6 regressor data
        analyze_exp_val_moe_regressors(
            quantile=quantile
        )
    analyze_exp_val_moe_classifiers(
        # Figure 6 classifier data
        exclude_timeout=True
    )


def reproduce_figure_7_results():
    time_limit_to_budget_map = {
        # Time limit: (time budget values in days, max time for an experiment to route)
        # Budget is in simulated time, not real world wall clock time
        1800: (5, 15, 25),
        3600: (10, 30, 50),
        5400: (25, 50, 75),
        7200: (30, 60, 90)
    }
    for filter_time_limit_key in time_limit_to_budget_map.keys():
        for route_time_budget_days in time_limit_to_budget_map[filter_time_limit_key]:
            route_time_budget_seconds = route_time_budget_days * 24*60*60
            perform_routing_sim_evaluations_fpl2025(
                target_predictor="ml-exp-val",
                naivety_constant=2.0,
                min_conf_to_extd_limit=0.5,
                route_time_budget=route_time_budget_seconds,
                filter_time_limit=filter_time_limit_key
            )
    for filter_time_limit_key in time_limit_to_budget_map.keys():
        for route_time_budget_days in time_limit_to_budget_map[filter_time_limit_key]:
            route_time_budget_seconds = route_time_budget_days * 24*60*60
            perform_routing_sim_evaluations_fpl2025(
                target_predictor="ml-iter",
                naivety_constant=0,
                min_conf_to_extd_limit=None,
                route_time_budget=route_time_budget_seconds,
                filter_time_limit=filter_time_limit_key
            )


def reproduce_figure_8_results():
    for filter_time_limit_val in (1800, 3600):
        for min_conf in (0.16, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8):
            perform_routing_sim_evaluations_fpl2025(
                target_predictor="ml-exp-val",
                naivety_constant=2.0,
                min_conf_to_extd_limit=min_conf,
                filter_time_limit=filter_time_limit_val
            )


def main():
    random_state = 1
    random.seed(random_state)
    reproduce_table_2_results(random_state=random_state)
    reproduce_figure_5_results(random_state=random_state)
    reproduce_figure_6_results()
    reproduce_figure_7_results()
    reproduce_figure_8_results()


if __name__ == "__main__":
    main()
