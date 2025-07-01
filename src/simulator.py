"""
For conducting all routing simulations
"""
from typing import Optional, Union
from predictors import SimulatorPredictorFPL
from constants import TimeIdx, LabelIdx, VTR_MAX_ITERS, AVG_SEC_PER_ITER


class RuleWindow:
    """
    This class is sued to require that a certain number of unroutable predictions are required within
        a window of N predictions in order to trigger early exit
    """
    def __init__(self, size, thresh):
        self.window = []
        self.size = size
        self.thresh = thresh
        if self.thresh == 0:
            self.thresh = 1

    def insert(self, iter_pred: bool):
        if self.size == 0:
            return
        assert isinstance(iter_pred, bool), \
            f"ERROR: Tried to insert non-bool into early exit window: {iter_pred}"
        if len(self.window) == self.size:
            del self.window[0]
        self.window.append(iter_pred)

    def should_ee(self):
        if sum(self.window) >= self.thresh:
            return True
        else:
            return False

    def reset(self):
        self.window = []


def sim_routing_attempt_fpl(
        X_class, X_reg, y, timing, init_time_limit: Union[int, float],
        sim_predictor: SimulatorPredictorFPL,
        window_size: Optional[int], totals, enable_ee: bool = True, iter_ee_mode: bool = False,
        is_naive_for_fast_routing: bool = False, naivety_constant: float = 0.0,
        exp_val_mode: bool = True, min_conf_to_extd_limit: Optional[float] = None
):
    """
    Simulate one routing attempt.
    :param X_class: Classification features
    :param X_reg: Regression features
    :param y: Labels
    :param timing: Simulated timing information (i.e. time taken to perform routing)
    :param init_time_limit: user-provided time limit
    :param sim_predictor: Predictor to employ for early exit
    :param window_size: Early exit window size
    :param totals: Label totals accumulated across routing
    :param enable_ee: Should early exit be enabled?
    :param iter_ee_mode: Should early exit be based on iteration predictions? Used for FCCM predictor.
    :param is_naive_for_fast_routing: Should we skip predictions when routing iterations occur quickly?
    :param naivety_constant: The amount by which a routing iteration is determined to be quick/slow.
    :param exp_val_mode: Early exit based on expected value of routing?
    :param min_conf_to_extd_limit: The minimum prediction confidence required to extend the user-provided time limit.
    :return:
    """
    assert len(X_class) == len(X_reg) == len(y) == len(timing)

    # CORE ROUTING HERE
    core_results = core_routing_fpl(
        X_class=X_class, X_reg=X_reg, y=y, timing=timing, init_time_limit=init_time_limit,
        extended_time_limit=2*init_time_limit, sim_predictor=sim_predictor,
        totals=totals, window_size=window_size,
        is_naive_for_fast_routing=is_naive_for_fast_routing, naivety_constant=naivety_constant,
        iter_ee_mode=iter_ee_mode, enable_ee=enable_ee, exp_val_mode=exp_val_mode,
        min_conf_to_extd_limit=min_conf_to_extd_limit
    )
    conv_pred_ee = None
    trav_limit_ee = None
    time_routed = core_results["time_routed"]
    time_ml = core_results["time_ml"]
    first_iter_time_ml = core_results["first_iter_time_ml"]
    is_routable = core_results["is_routable"]
    routed_to_end = core_results["routed_to_end"]
    iters_routed = core_results["iters_routed"]
    routed_all_iters = core_results["routed_all_iters"]

    if time_routed < timing[0][TimeIdx.ITER_PURE_ROUTE_TIME]:
        # If this is true, then that means we simulated the first iteration of routing and when we checked time_routed,
        # it was greater than the time limit, so we reset time_routed to the time limit.
        # We would have terminated routing prior to completing a single iteration in real routing
        # Just set to zero, effectively excluding this run.
        true_time_routed = 0
    else:
        true_time_routed = time_routed - timing[0][TimeIdx.ITER_PURE_ROUTE_TIME]
    true_time_ml = time_ml - first_iter_time_ml
    assert true_time_ml >= 0, f"{true_time_ml} >= 0"

    results = {
        "is_routable": is_routable,
        "routed_to_end": routed_to_end,
        "iters_routed": iters_routed,
        "time_routed": time_routed,
        "time_ml": time_ml,
        "true_time_routed": true_time_routed,
        "true_time_ml": true_time_ml,
        "conv_pred_ee": conv_pred_ee,
        "trav_limit_ee": trav_limit_ee,
        "routed_all_iters": routed_all_iters
    }
    return results


def core_routing_fpl(
        X_class, X_reg, y, timing,
        init_time_limit: Union[int, float], extended_time_limit: Union[int, float],
        totals, window_size: int, is_naive_for_fast_routing: bool, naivety_constant: float,
        iter_ee_mode: bool, enable_ee: bool,
        sim_predictor: Optional[SimulatorPredictorFPL],
        exp_val_mode: bool, min_conf_to_extd_limit: Optional[float],
):
    """
    Core routing simulation routine that has a flexible time limit. Operates in two modes:
    1) Expected-value-based routing. Predictor decides whether to continue routing based on the expected
        likelihood of routing success, factoring in probability of convergence and finishing within time limit.
    2) Pure expectation routing (i.e. keep routing if likelihood of convergence > 0.5 and predicted runtime
        < time limit). Use an extended time limit if likelihood of convergence is very high.
    :return:
    """
    assert (sim_predictor is not None) or (not enable_ee)
    assert not (exp_val_mode and iter_ee_mode)
    if enable_ee:
        if iter_ee_mode:
            assert min_conf_to_extd_limit is None
        else:
            assert min_conf_to_extd_limit is not None
    assert extended_time_limit > init_time_limit
    assert extended_time_limit <= 14400  # 4 hours
    # Metrics to track:
    # 1) Is this circuit routable?
    # 2) Did we successfully route the circuit?
    # 3) Iters routed
    # 4) Routing time
    # 5) ML time
    # 6-7) "True" versions of 4-5 (i.e. ignore first iteration, don't need true iters as this is normal iters - 1)
    iters_routed = 0
    time_routed = 0
    time_ml = 0
    first_iter_time_ml = 0

    # Determining routability:
    # The circuit must complete routing within the provided time limit and the global routing limit of 1000 iterations.
    # First we'll check if the circuit completed routing at all during initial data extraction.
    # This is denoted by positive labels in the simulation data, negative labels indicated failed routing.
    # If the labels are negative, we know the answer.
    # Otherwise, we have to check if the routing exceeds the provided time limit.
    if y[0][0] < 0:
        assert totals[0] < 0, f"{totals[0]} < 0"  # A heuristic error check
        is_routable_init = False
        is_routable_extd = False
    else:
        assert totals[LabelIdx.TOT_PURE_ROUTE_TIME] > 0, f"{totals[LabelIdx.TOT_PURE_ROUTE_TIME] > 0}"
        assert totals[LabelIdx.ROUTE_ITER] > 0, f"{totals[LabelIdx.ROUTE_ITER] > 0}"
        if totals[LabelIdx.ROUTE_ITER] > VTR_MAX_ITERS:
            is_routable_init = False
            is_routable_extd = False
        else:
            if totals[LabelIdx.TOT_PURE_ROUTE_TIME] > extended_time_limit:
                is_routable_init = False
                is_routable_extd = False
            elif totals[LabelIdx.TOT_PURE_ROUTE_TIME] > init_time_limit:
                is_routable_init = False
                is_routable_extd = True
            else:
                is_routable_init = True
                is_routable_extd = True

    if iter_ee_mode:
        # Early exit based on iteration prediction
        # All other aspects (e.g. ending at time_limit) remain unchanged
        # Now we just use iteration predictions to power early exit
        iter_limit = min(
            VTR_MAX_ITERS,
            int(round(init_time_limit/AVG_SEC_PER_ITER))
        )
    else:
        iter_limit = 9999

    if sim_predictor is not None:
        target_label_reg = sim_predictor.target_label_regression
    else:
        target_label_reg = LabelIdx.NODE_TRAV
    target_label_total = totals[target_label_reg]
    # (use absolute value below because remaining amounts could be negative if the true amounts are unknown)
    prev_iter_workload_remaining = abs(target_label_total)  # total - amount remaining after iter idx 0
    ee_window = RuleWindow(size=window_size, thresh=window_size)
    ee_window.reset()
    routed_to_end = True  # Assume True and prove False if we early exit
    routed_all_iters = False  # Assume False and prove True if we don't early exit
    total_num_iters = len(y)
    active_time_limit = init_time_limit
    time_limit_was_extended = False
    for iter_idx in range(total_num_iters):
        # Get data for this iteration
        X_class_row = X_class[[iter_idx], :]
        X_reg_row = X_reg[[iter_idx], :]
        y_row = y[iter_idx]
        timing_row = timing[iter_idx]
        # Get routing time for this iteration
        iter_route_time = timing_row[TimeIdx.ITER_PURE_ROUTE_TIME]
        time_routed += iter_route_time
        iters_routed += 1

        # Get workload handled this iteration
        # (use absolute value below because remaining amounts could be negative if the true amounts are unknown)
        iter_workload_remaining = abs(y_row[target_label_reg])
        assert iter_workload_remaining < prev_iter_workload_remaining, \
            f"{iter_workload_remaining} < {prev_iter_workload_remaining}"
        iter_workload = prev_iter_workload_remaining - iter_workload_remaining

        # Check if we reached the end of routing
        if iters_routed == total_num_iters:
            if iter_idx == 0:
                print("WARNING: RUN ONLY HAD A SINGLE ITERATION")
            routed_to_end = True
            routed_all_iters = True
            break
        elif (time_routed > init_time_limit and not time_limit_was_extended) or \
                (time_routed > extended_time_limit and time_limit_was_extended):
            if iter_idx == 0:
                print("WARNING: RUN ONLY HAD A SINGLE ITERATION")
            time_routed = active_time_limit  # Adjust to match reality (would kill route midway through iteration)
            routed_to_end = True
            break

        # Check if we should bother predicting
        if (is_naive_for_fast_routing and iter_route_time < naivety_constant) or not enable_ee:
            prev_iter_workload_remaining = iter_workload_remaining
            continue

        # Get predictions
        # (convergence probability, workload remaining, prediction time, expected value)
        # Note that these can be None, except prediction time which would be 0 for no prediction
        iter_route_speed = iter_workload / iter_route_time  # If the workload is time, this resolves to unity
        convergence_proba, workload_remaining_pred, pred_time, trav_completion_proba = sim_predictor.sim_predict(
            X_row_class=X_class_row, X_row_reg=X_reg_row, timing_row=timing_row,
            time_remaining_to_limit=(extended_time_limit-time_routed), iter_route_speed=iter_route_speed,
            num_iters_routed=iters_routed, time_routed=time_routed
        )
        time_ml += pred_time  # This already factors feature extraction + inference time

        if convergence_proba is None:
            # We did not make any prediction, default to continue routing
            prev_iter_workload_remaining = iter_workload_remaining
            continue

        # Determine if we should place an early exit vote or a continued routing vote
        if exp_val_mode:
            # Expected value routing (product of probabilities)
            if convergence_proba * trav_completion_proba >= min_conf_to_extd_limit:
                # Require very high confidence to extend time limit
                active_time_limit = extended_time_limit
                time_limit_was_extended = True  # Permanently change the timeout condition
            else:
                active_time_limit = init_time_limit  # Predict against the lower time limit, but keep extended timeout
        elif not iter_ee_mode:
            # Pure expectation mode
            if convergence_proba >= min_conf_to_extd_limit:
                # Require very high confidence to extend time limit
                active_time_limit = extended_time_limit
            else:
                active_time_limit = init_time_limit
        if convergence_proba >= 0.5:
            # Correct rare edge cases where models can predict a small negative value
            if workload_remaining_pred is not None:  # We may not always have predicted this quantity
                if workload_remaining_pred < 0:
                    workload_remaining_pred = 1
            # Make decision based on prediction
            if iter_ee_mode:
                # We have predicted the circuit will route eventually (within 1000 iterations)
                # Predict iterations remaining and compare against iteration limit
                assert workload_remaining_pred is not None  # Should be true if we predicted convergence
                iters_remaining_pred = workload_remaining_pred
                if iters_routed + iters_remaining_pred > iter_limit:
                    should_ee_pred = True
                else:
                    should_ee_pred = False
            else:
                # We have predicted the circuit will route eventually (within 1000 iterations)
                # Convert workload prediction to a time prediction
                # (unless predictor directly predicts time)
                time_remaining_pred = workload_remaining_pred / iter_route_speed
                if time_routed + time_remaining_pred > active_time_limit:
                    should_ee_pred = True
                else:
                    should_ee_pred = False
        else:
            should_ee_pred = True
        ee_window.insert(should_ee_pred)

        prev_iter_workload_remaining = iter_workload_remaining

        # Should we early exit?
        if ee_window.should_ee():
            if enable_ee:
                routed_to_end = False
                break

    if iter_ee_mode or not enable_ee:
        is_routable = is_routable_init
    else:
        is_routable = is_routable_extd

    return {
        "time_routed": time_routed,
        "first_iter_time_ml": first_iter_time_ml,
        "time_ml": time_ml,
        "is_routable": is_routable,
        "routed_to_end": routed_to_end,
        "routed_all_iters": routed_all_iters,
        "iters_routed": iters_routed
    }
