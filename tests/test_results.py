# test_results.py
import pytest

from pycmtensor.statistics import elasticities


@pytest.fixture(scope="module")
def trained_model(swissmetro_db, mnl_model):
    model = mnl_model
    db = swissmetro_db
    model.config.set_hyperparameter("max_steps", 10)
    model.train(db)
    return model


def test_model(mnl_model):
    m = mnl_model
    m.reset_values()
    assert isinstance(m.get_betas(), dict)
    assert len(m.get_weights()) == 0
    assert str(m) == "MNL"


def test_beta_statistics(trained_model):
    model = trained_model
    assert len(model.results.beta_statistics()) == 5


def test_model_statistics(trained_model):
    model = trained_model
    assert len(model.results.model_statistics()) == 11


def test_benchmark(trained_model):
    model = trained_model
    assert len(model.results.benchmark()) == 4


def test_model_correlation_matrix(trained_model):
    model = trained_model
    assert len(model.results.model_correlation_matrix()) == 4
    assert len(model.results.model_correlation_matrix().columns) == 4


def test_model_robust_correlation_matrix(trained_model):
    model = trained_model
    assert len(model.results.model_robust_correlation_matrix()) == 4
    assert len(model.results.model_robust_correlation_matrix().columns) == 4


def test_prediction(trained_model, swissmetro_db):
    model = trained_model
    db = swissmetro_db
    prediction = model.predict(db, return_choices=False)
    assert prediction.shape == (10719, 3)
    choices = model.predict(db, return_choices=True)
    assert choices.shape == (10719,)


def test_elasticities(trained_model, swissmetro_db):
    model = trained_model
    db = swissmetro_db
    e = elasticities(model, db, 0, "TRAIN_TT")
    assert len(e) == 8575
