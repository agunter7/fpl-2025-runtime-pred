from typing import Union, Optional, Tuple
import numpy as np
from constants import LabelIdx, TimeIdx


class CustomEstimator:
    def __init__(self,
                 name: str,
                 is_classifier: bool,
                 estimator=None,
                 feature_maximums: Optional[np.ndarray] = None,
                 feature_minimums: Optional[np.ndarray] = None
                 ):
        self.name = name
        self.estimator = estimator
        self.is_regressor = is_classifier
        self.feature_maximums = feature_maximums
        self.feature_minimums = feature_minimums


class CustomClassifier(CustomEstimator):
    def __init__(
            self, name: str, estimator,
            feature_maximums: Optional[np.ndarray] = None,
            feature_minimums: Optional[np.ndarray] = None
    ):
        super().__init__(
            name=name, is_classifier=True, estimator=estimator,
            feature_maximums=feature_maximums, feature_minimums=feature_minimums
        )

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict_pos_proba_row(self, X_row):
        return self.estimator.predict_proba(X_row)[0][1]


class CustomRegressor(CustomEstimator):
    def __init__(
            self, name: str, estimator,
            feature_maximums: Optional[np.ndarray] = None,
            feature_minimums: Optional[np.ndarray] = None
    ):
        super().__init__(
            name=name, is_classifier=False, estimator=estimator,
            feature_maximums=feature_maximums, feature_minimums=feature_minimums
        )

    def predict(self, X):
        return np.abs(self.estimator.predict(X))

    def predict_row(self, X_row):
        return np.abs(self.estimator.predict(X_row))[0]


class SimulatorPredictorFCCM:
    """
    Predictor from FCCM 2023 Paper:
    A machine learning approach for predicting the difficulty of fpga routing problems
    """
    target_label_regression = LabelIdx.ROUTE_ITER

    def __init__(
            self,
            name: str,
            f1000, f400, f250, f150,
            r1000, r400, r250, r150,
            use_overuse_features: bool, use_switchbox_features: bool,
            use_wlpa_features: bool, use_ncpr_features: bool,
            explicitly_override_extraction_time: bool = False
    ):
        self.name = name
        self.f1000 = f1000
        self.f400 = f400
        self.f250 = f250
        self.f150 = f150
        self.r1000 = r1000
        self.r400 = r400
        self.r250 = r250
        self.r150 = r150
        self.use_overuse_features = use_overuse_features
        self.use_switchbox_features = use_switchbox_features
        self.use_wlpa_features = use_wlpa_features
        self.use_ncpr_features = use_ncpr_features
        for feature_set in (use_overuse_features, use_switchbox_features, use_wlpa_features, use_ncpr_features):
            if feature_set is False and not explicitly_override_extraction_time:
                print("WARNING: FCCM MoE IS NOT USING ALL FEATURE EXTRACTION TIME VALUES")

    def sim_predict(
            self, X_row_class, X_row_reg, timing_row, time_remaining_to_limit, iter_route_speed,
            num_iters_routed, time_routed
    ) -> Tuple[Union[int, bool], Optional[Union[int, float]], float, Optional[float]]:
        # Get route convergence prediction
        is_convergent = self.f1000.estimator.predict(X_row_class)[0]
        prediction_time = 0
        if self.use_overuse_features:
            prediction_time += timing_row[TimeIdx.OVER]
        if self.use_switchbox_features:
            prediction_time += timing_row[TimeIdx.SB]
        if self.use_wlpa_features:
            prediction_time += timing_row[TimeIdx.WLPA]
        if self.use_ncpr_features:
            prediction_time += timing_row[TimeIdx.NCPR]

        if is_convergent == 0:
            return 0, None, prediction_time, None

        # Get expected route iters remaining
        if self.f150.estimator.predict(X_row_class)[0] == 1:
            regressor = self.r150
        elif self.f250.estimator.predict(X_row_class)[0] == 1:
            regressor = self.r250
        elif self.f400.estimator.predict(X_row_class)[0] == 1:
            regressor = self.r400
        elif is_convergent:
            regressor = self.r1000
        else:
            regressor = None
            exit("ERROR: Bad regressor selection condition")

        expected_iters_remaining = np.abs(regressor.estimator.predict(X_row_reg))[0]

        return is_convergent, expected_iters_remaining, prediction_time, None


class SimulatorPredictorFPL:
    """
    Predictor from FPL 2025 Paper:
    Open-Source FPGA Routing Runtime Prediction for Improved Productivity via Smart Route Termination
    """
    def __init__(
            self, name: str, classifier: CustomClassifier, mean_regressor: CustomRegressor,
            target_label_regression: LabelIdx, use_overuse_features: bool,
            use_switchbox_features: bool, use_wlpa_features: bool,
            use_ncpr_features: bool, q5, q10, q20, q30, q40, q50, q60, q70, q80, q90, q95
    ):
        self.name = name
        self.target_label_regression = target_label_regression
        self.classifier = classifier
        self.mean_regressor = mean_regressor
        self.quantile_predictors = (
            (5, q5),
            (10, q10),
            (20, q20),
            (30, q30),
            (40, q40),
            (50, q50),
            (60, q60),
            (70, q70),
            (80, q80),
            (90, q90),
            (95, q95)
        )
        self.use_overuse_features = use_overuse_features
        self.use_switchbox_features = use_switchbox_features
        self.use_wlpa_features = use_wlpa_features
        self.use_ncpr_features = use_ncpr_features

    def sim_predict(
            self, X_row_class, X_row_reg, timing_row, time_remaining_to_limit, iter_route_speed,
            num_iters_routed, time_routed
    ) -> Tuple[bool, Optional[Union[int, float]], float, float]:
        # Get probability of routing convergence
        convergence_proba = self.classifier.estimator.predict_proba(X_row_class)[0][1]

        # Get expected workload remaining
        expected_route_completion_workload = np.abs(self.mean_regressor.estimator.predict(X_row_reg))[0]

        # Get probability of routing completing within the remaining time before hitting the time limit
        trav_limit = iter_route_speed*time_remaining_to_limit
        prev_quantile = 0
        prev_quantile_prediction = 0
        traversal_completion_proba = None
        for quantile, quantile_predictor in self.quantile_predictors:
            quantile_prediction = np.abs(quantile_predictor.estimator.predict(X_row_reg))[0]
            if trav_limit <= quantile_prediction:
                if quantile == 5:
                    # Just say anything below 5th quantile is zero
                    traversal_completion_proba = 0
                else:
                    # Linearly interpolate between this and the last quantile
                    dist_to_prev_quantile_pred = trav_limit - prev_quantile_prediction
                    interquantile_pred = quantile_prediction - prev_quantile_prediction
                    interquantile_dist = quantile - prev_quantile
                    interquantile_gradient = interquantile_dist / interquantile_pred
                    interpolated_quantile = prev_quantile + \
                                            (dist_to_prev_quantile_pred * interquantile_gradient)
                    traversal_completion_proba = interpolated_quantile/100
                break
            prev_quantile = quantile
            prev_quantile_prediction = quantile_prediction
        if traversal_completion_proba is None:
            # Value mapped beyond 95th quantile, just say it is guaranteed
            traversal_completion_proba = 1

        # Determine how much time this prediction took (extraction + inference)
        prediction_time = 0
        if self.use_overuse_features:
            prediction_time += timing_row[TimeIdx.OVER]
        if self.use_switchbox_features:
            prediction_time += timing_row[TimeIdx.SB]
        if self.use_wlpa_features:
            prediction_time += timing_row[TimeIdx.WLPA]
        if self.use_ncpr_features:
            prediction_time += timing_row[TimeIdx.NCPR]

        return convergence_proba, expected_route_completion_workload, prediction_time, traversal_completion_proba
