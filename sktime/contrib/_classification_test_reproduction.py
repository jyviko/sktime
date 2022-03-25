# -*- coding: utf-8 -*-
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.dictionary_based import (
    MUSE,
    WEASEL,
    BOSSEnsemble,
    ContractableBOSS,
    TemporalDictionaryEnsemble,
)
from sktime.classification.distance_based import (
    ElasticEnsemble,
    ProximityForest,
    ShapeDTW,
)
from sktime.classification.early_classification import (
    TEASER,
    ProbabilityThresholdEarlyClassifier,
)
from sktime.classification.feature_based import (
    Catch22Classifier,
    FreshPRINCE,
    MatrixProfileClassifier,
    RandomIntervalClassifier,
    SignatureClassifier,
    SummaryClassifier,
    TSFreshClassifier,
)
from sktime.classification.hybrid import HIVECOTEV1, HIVECOTEV2
from sktime.classification.interval_based import (
    CanonicalIntervalForest,
    DrCIF,
    RandomIntervalSpectralEnsemble,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)
from sktime.classification.kernel_based import Arsenal, RocketClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.datasets import load_basic_motions, load_unit_test
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from sktime.transformations.panel.catch22 import Catch22
from sktime.transformations.panel.random_intervals import RandomIntervals
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform
from sktime.transformations.series.summarize import SummaryTransformer


def _reproduce_classification_unit_test(estimator):
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    estimator.fit(X_train, y_train)
    return estimator.predict_proba(X_test.iloc[indices])


def _reproduce_classification_basic_motions(estimator):
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    estimator.fit(X_train.iloc[indices], y_train[indices])
    return estimator.predict_proba(X_test.iloc[indices])


def _reproduce_early_classification_unit_test(estimator):
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    estimator.fit(X_train, y_train)

    final_probas = np.zeros((10, 2))
    final_decisions = np.zeros(10)

    X_test = from_nested_to_3d_numpy(X_test)
    states = None
    for i in estimator.classification_points:
        X = X_test[indices, :, :i]
        probas, decisions, states = estimator.predict_proba(X, state_info=states)

        for n in range(10):
            if decisions[n] and final_decisions[n] == 0:
                final_probas[n] = probas[n]
                final_decisions[n] = i

    return final_probas


def _reproduce_transform_unit_test(estimator):
    X_train, y_train = load_unit_test(split="train")
    indices = np.random.RandomState(0).choice(len(y_train), 5, replace=False)

    estimator.fit(X_train.iloc[indices], y_train[indices])
    return np.nan_to_num(estimator.transform(X_train.iloc[indices]), False, 0, 0, 0)


def _reproduce_transform_basic_motions(estimator):
    X_train, y_train = load_basic_motions(split="train")
    indices = np.random.RandomState(4).choice(len(y_train), 5, replace=False)

    estimator.fit(X_train.iloc[indices], y_train[indices])
    return np.nan_to_num(estimator.transform(X_train.iloc[indices]), False, 0, 0, 0)


def _print_array(test_name, array):
    print(test_name)
    print("[")
    for sub_array in array:
        print("[", end="")
        for i, value in enumerate(sub_array):
            print(value.astype(str), end="")
            if i < len(sub_array) - 1:
                print(", ", end="")
        print("],")
    print("]")


if __name__ == "__main__":
    _print_array(
        "ColumnEnsembleClassifier - BasicMotions",
        _reproduce_classification_basic_motions(
            ColumnEnsembleClassifier(
                estimators=[
                    (
                        "cBOSS",
                        ContractableBOSS(
                            n_parameter_samples=4, max_ensemble_size=2, random_state=0
                        ),
                        [5],
                    ),
                    (
                        "CIF",
                        CanonicalIntervalForest(
                            n_estimators=2,
                            n_intervals=4,
                            att_subsample_size=4,
                            random_state=0,
                        ),
                        [3, 4],
                    ),
                ]
            )
        ),
    )
    _print_array(
        "BOSSEnsemble - UnitTest",
        _reproduce_classification_unit_test(
            BOSSEnsemble(max_ensemble_size=5, random_state=0)
        ),
    )
    _print_array(
        "ContractableBOSS - UnitTest",
        _reproduce_classification_unit_test(
            ContractableBOSS(
                n_parameter_samples=10, max_ensemble_size=5, random_state=0
            )
        ),
    )
    _print_array(
        "MUSE - BasicMotions",
        _reproduce_classification_basic_motions(
            MUSE(window_inc=4, use_first_order_differences=False, random_state=0)
        ),
    )
    _print_array(
        "TemporalDictionaryEnsemble - UnitTest",
        _reproduce_classification_unit_test(
            TemporalDictionaryEnsemble(
                n_parameter_samples=10,
                max_ensemble_size=5,
                randomly_selected_params=5,
                random_state=0,
            )
        ),
    )
    _print_array(
        "TemporalDictionaryEnsemble - BasicMotions",
        _reproduce_classification_basic_motions(
            TemporalDictionaryEnsemble(
                n_parameter_samples=10,
                max_ensemble_size=5,
                randomly_selected_params=5,
                random_state=0,
            )
        ),
    )
    _print_array(
        "WEASEL - UnitTest",
        _reproduce_classification_unit_test(WEASEL(window_inc=4, random_state=0)),
    )
    _print_array(
        "ElasticEnsemble - UnitTest",
        _reproduce_classification_unit_test(
            ElasticEnsemble(
                proportion_of_param_options=0.1,
                proportion_train_for_test=0.1,
                majority_vote=True,
                distance_measures=["dtw", "ddtw"],
                random_state=0,
            )
        ),
    )
    _print_array(
        "ProximityForest - UnitTest",
        _reproduce_classification_unit_test(
            ProximityForest(n_estimators=5, random_state=0)
        ),
    )
    _print_array("ShapeDTW - UnitTest", _reproduce_classification_unit_test(ShapeDTW()))
    _print_array(
        "Catch22Classifier - UnitTest",
        _reproduce_classification_unit_test(
            Catch22Classifier(
                estimator=RandomForestClassifier(n_estimators=10),
                outlier_norm=True,
                random_state=0,
            )
        ),
    )
    _print_array(
        "Catch22Classifier - BasicMotions",
        _reproduce_classification_basic_motions(
            Catch22Classifier(
                estimator=RandomForestClassifier(n_estimators=10),
                outlier_norm=True,
                random_state=0,
            )
        ),
    )
    _print_array(
        "FreshPRINCE - UnitTest",
        _reproduce_classification_unit_test(
            FreshPRINCE(
                default_fc_parameters="minimal",
                n_estimators=10,
                random_state=0,
            )
        ),
    )
    _print_array(
        "MatrixProfileClassifier - UnitTest",
        _reproduce_classification_unit_test(MatrixProfileClassifier(random_state=0)),
    )
    _print_array(
        "RandomIntervalClassifier - UnitTest",
        _reproduce_classification_unit_test(
            RandomIntervalClassifier(
                n_intervals=3,
                interval_transformers=SummaryTransformer(
                    summary_function=("mean", "std", "min", "max"),
                    quantiles=(0.25, 0.5, 0.75),
                ),
                estimator=RandomForestClassifier(n_estimators=10),
                random_state=0,
            )
        ),
    )
    _print_array(
        "RandomIntervalClassifier - BasicMotions",
        _reproduce_classification_basic_motions(
            RandomIntervalClassifier(
                n_intervals=3,
                interval_transformers=SummaryTransformer(
                    summary_function=("mean", "std", "min", "max"),
                    quantiles=(0.25, 0.5, 0.75),
                ),
                estimator=RandomForestClassifier(n_estimators=10),
                random_state=0,
            )
        ),
    )
    _print_array(
        "SignatureClassifier - UnitTest",
        _reproduce_classification_unit_test(
            SignatureClassifier(
                estimator=RandomForestClassifier(n_estimators=10), random_state=0
            )
        ),
    )
    _print_array(
        "SignatureClassifier - BasicMotions",
        _reproduce_classification_basic_motions(
            SignatureClassifier(
                estimator=RandomForestClassifier(n_estimators=10), random_state=0
            )
        ),
    )
    _print_array(
        "SummaryClassifier - UnitTest",
        _reproduce_classification_unit_test(
            SummaryClassifier(
                estimator=RandomForestClassifier(n_estimators=10), random_state=0
            )
        ),
    )
    _print_array(
        "SummaryClassifier - BasicMotions",
        _reproduce_classification_basic_motions(
            SummaryClassifier(
                estimator=RandomForestClassifier(n_estimators=10), random_state=0
            )
        ),
    )
    _print_array(
        "TSFreshClassifier - UnitTest",
        _reproduce_classification_unit_test(
            TSFreshClassifier(
                default_fc_parameters="minimal",
                relevant_feature_extractor=False,
                estimator=RandomForestClassifier(n_estimators=10),
                random_state=0,
            )
        ),
    )
    _print_array(
        "TSFreshClassifier - BasicMotions",
        _reproduce_classification_basic_motions(
            TSFreshClassifier(
                default_fc_parameters="minimal",
                relevant_feature_extractor=False,
                estimator=RandomForestClassifier(n_estimators=10),
                random_state=0,
            )
        ),
    )
    _print_array(
        "HIVECOTEV1 - UnitTest",
        _reproduce_classification_unit_test(
            HIVECOTEV1(
                stc_params={
                    "estimator": RandomForestClassifier(n_estimators=3),
                    "n_shapelet_samples": 50,
                    "max_shapelets": 5,
                    "batch_size": 10,
                },
                tsf_params={"n_estimators": 3},
                rise_params={"n_estimators": 3},
                cboss_params={"n_parameter_samples": 5, "max_ensemble_size": 3},
                random_state=0,
            )
        ),
    )
    _print_array(
        "HIVECOTEV2 - UnitTest",
        _reproduce_classification_unit_test(
            HIVECOTEV2(
                stc_params={
                    "estimator": RandomForestClassifier(n_estimators=3),
                    "n_shapelet_samples": 50,
                    "max_shapelets": 5,
                    "batch_size": 10,
                },
                drcif_params={
                    "n_estimators": 3,
                    "n_intervals": 2,
                    "att_subsample_size": 2,
                },
                arsenal_params={"num_kernels": 50, "n_estimators": 3},
                tde_params={
                    "n_parameter_samples": 5,
                    "max_ensemble_size": 3,
                    "randomly_selected_params": 3,
                },
                random_state=0,
            )
        ),
    )
    _print_array(
        "HIVECOTEV2 - BasicMotions",
        _reproduce_classification_basic_motions(
            HIVECOTEV2(
                stc_params={
                    "estimator": RandomForestClassifier(n_estimators=3),
                    "n_shapelet_samples": 50,
                    "max_shapelets": 5,
                    "batch_size": 10,
                },
                drcif_params={
                    "n_estimators": 3,
                    "n_intervals": 2,
                    "att_subsample_size": 2,
                },
                arsenal_params={"num_kernels": 50, "n_estimators": 3},
                tde_params={
                    "n_parameter_samples": 5,
                    "max_ensemble_size": 3,
                    "randomly_selected_params": 3,
                },
                random_state=0,
            )
        ),
    )
    _print_array(
        "CanonicalIntervalForest - UnitTest",
        _reproduce_classification_unit_test(
            CanonicalIntervalForest(
                n_estimators=10, n_intervals=2, att_subsample_size=4, random_state=0
            )
        ),
    )
    _print_array(
        "CanonicalIntervalForest - BasicMotions",
        _reproduce_classification_basic_motions(
            CanonicalIntervalForest(
                n_estimators=10, n_intervals=2, att_subsample_size=4, random_state=0
            )
        ),
    )
    _print_array(
        "DrCIF - UnitTest",
        _reproduce_classification_unit_test(
            DrCIF(n_estimators=10, n_intervals=2, att_subsample_size=4, random_state=0)
        ),
    )
    _print_array(
        "DrCIF - BasicMotions",
        _reproduce_classification_basic_motions(
            DrCIF(n_estimators=10, n_intervals=2, att_subsample_size=4, random_state=0)
        ),
    )
    _print_array(
        "RandomIntervalSpectralEnsemble - UnitTest",
        _reproduce_classification_unit_test(
            RandomIntervalSpectralEnsemble(n_estimators=10, random_state=0)
        ),
    )
    _print_array(
        "SupervisedTimeSeriesForest - UnitTest",
        _reproduce_classification_unit_test(
            SupervisedTimeSeriesForest(n_estimators=10, random_state=0)
        ),
    )
    _print_array(
        "TimeSeriesForestClassifier - UnitTest",
        _reproduce_classification_unit_test(
            TimeSeriesForestClassifier(n_estimators=10, random_state=0)
        ),
    )
    _print_array(
        "Arsenal - UnitTest",
        _reproduce_classification_unit_test(
            Arsenal(num_kernels=200, n_estimators=5, random_state=0)
        ),
    )
    _print_array(
        "Arsenal - BasicMotions",
        _reproduce_classification_basic_motions(
            Arsenal(num_kernels=200, n_estimators=5, random_state=0)
        ),
    )
    _print_array(
        "RocketClassifier - UnitTest",
        _reproduce_classification_unit_test(
            RocketClassifier(num_kernels=500, random_state=0)
        ),
    )
    _print_array(
        "RocketClassifier - BasicMotions",
        _reproduce_classification_basic_motions(
            RocketClassifier(num_kernels=500, random_state=0)
        ),
    )
    _print_array(
        "ShapeletTransformClassifier - UnitTest",
        _reproduce_classification_unit_test(
            ShapeletTransformClassifier(
                estimator=RotationForest(n_estimators=4),
                max_shapelets=30,
                n_shapelet_samples=200,
                batch_size=50,
                random_state=0,
            )
        ),
    )
    _print_array(
        "ShapeletTransformClassifier - BasicMotions",
        _reproduce_classification_basic_motions(
            ShapeletTransformClassifier(
                estimator=RotationForest(n_estimators=4),
                max_shapelets=30,
                n_shapelet_samples=200,
                batch_size=50,
                random_state=0,
            )
        ),
    )

    _print_array(
        "ProbabilityThresholdEarlyClassifier - UnitTest",
        _reproduce_early_classification_unit_test(
            ProbabilityThresholdEarlyClassifier(
                random_state=0,
                classification_points=[6, 16, 24],
                probability_threshold=1,
                estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
            )
        ),
    )
    _print_array(
        "TEASER - UnitTest",
        _reproduce_early_classification_unit_test(
            TEASER(
                random_state=0,
                classification_points=[6, 10, 16, 24],
                estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
            )
        ),
    )
    _print_array(
        "TEASER-IF - UnitTest",
        _reproduce_early_classification_unit_test(
            TEASER(
                random_state=0,
                classification_points=[6, 10, 16, 24],
                estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
                one_class_classifier=IsolationForest(n_estimators=5),
                one_class_param_grid={"bootstrap": [True, False]},
            )
        ),
    )

    _print_array(
        "Catch22 - UnitTest",
        _reproduce_transform_unit_test(Catch22(outlier_norm=True)),
    )
    _print_array(
        "Catch22 - BasicMotions",
        _reproduce_transform_basic_motions(Catch22()),
    )
    _print_array(
        "RandomIntervals - UnitTest",
        _reproduce_transform_unit_test(RandomIntervals(random_state=0, n_intervals=3)),
    )
    _print_array(
        "RandomIntervals - BasicMotions",
        _reproduce_transform_basic_motions(
            RandomIntervals(random_state=0, n_intervals=3)
        ),
    )
    _print_array(
        "RandomShapeletTransform - UnitTest",
        _reproduce_transform_unit_test(
            RandomShapeletTransform(
                max_shapelets=10, n_shapelet_samples=500, random_state=0
            )
        ),
    )
    _print_array(
        "RandomShapeletTransform - BasicMotions",
        _reproduce_transform_basic_motions(
            RandomShapeletTransform(
                max_shapelets=10, n_shapelet_samples=500, random_state=0
            )
        ),
    )
