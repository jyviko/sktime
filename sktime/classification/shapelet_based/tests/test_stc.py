# -*- coding: utf-8 -*-
"""ShapeletTransformClassifier test code."""
import numpy as np
from numpy import testing
from sklearn.metrics import accuracy_score

from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.datasets import load_basic_motions, load_unit_test


def test_stc_on_unit_test_data():
    """Test of ShapeletTransformClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    X_test, y_test = load_unit_test(split="test")
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # train STC
    stc = ShapeletTransformClassifier(
        estimator=RotationForest(n_estimators=4),
        max_shapelets=30,
        n_shapelet_samples=200,
        batch_size=50,
        random_state=0,
        save_transformed_data=True,
    )
    stc.fit(X_train, y_train)

    # assert probabilities are the same
    probas = stc.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, stc_unit_test_probas, decimal=2)

    # test train estimate
    train_probas = stc._get_train_probs(X_train, y_train)
    train_preds = stc.classes_[np.argmax(train_probas, axis=1)]
    assert accuracy_score(y_train, train_preds) >= 0.75


def test_contracted_stc_on_unit_test_data():
    """Test of contracted ShapeletTransformClassifier on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")

    # train contracted STC
    stc = ShapeletTransformClassifier(
        estimator=RotationForest(contract_max_n_estimators=4),
        max_shapelets=30,
        time_limit_in_minutes=0.25,
        contract_max_n_shapelet_samples=200,
        batch_size=50,
        random_state=0,
    )
    stc.fit(X_train, y_train)


def test_stc_on_basic_motions():
    """Test of ShapeletTransformClassifier on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # train STC
    stc = ShapeletTransformClassifier(
        estimator=RotationForest(n_estimators=4),
        max_shapelets=30,
        n_shapelet_samples=200,
        batch_size=50,
        random_state=0,
    )
    stc.fit(X_train.iloc[indices], y_train[indices])

    # assert probabilities are the same
    probas = stc.predict_proba(X_test.iloc[indices])
    testing.assert_array_almost_equal(probas, stc_basic_motions_probas, decimal=2)


stc_unit_test_probas = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.75, 0.25],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.75, 0.25],
    ]
)
stc_basic_motions_probas = np.array(
    [
        [0.0, 0.0, 0.25, 0.75],
        [0.75, 0.25, 0.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.5, 0.25, 0.25, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.5, 0.25, 0.0, 0.25],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
