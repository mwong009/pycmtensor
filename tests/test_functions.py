# test_functions.py
import aesara
import aesara.tensor as aet
import numpy as np
import pytest

import pycmtensor.functions as functions


def test_relu():
    x = aet.scalar("x")
    y = functions.relu(x).eval({x: 0.6})
    # test regular relu
    assert functions.relu(x).eval({x: 0.6}) == 0.6
    assert functions.relu(x).eval({x: -0.8}) == 0.0

    # linear
    assert functions.relu(x, alpha=1.0).eval({x: 1.0}) == 1.0
    assert functions.relu(x, alpha=1.0).eval({x: -3.0}) == -3.0


def test_exp_mov_average():
    batch_avg = aet.vector("batch_avg")
    moving_avg = aet.scalar("moving_avg")

    y = functions.exp_mov_average(batch_avg=batch_avg, moving_avg=moving_avg, alpha=0.1)

    out = y.eval({batch_avg: np.array([0.2, 0.3, 0.4]), moving_avg: np.array(0.1)})

    assert out.shape[0] == 3
    assert np.allclose(out, np.array([0.11, 0.12, 0.13]))


@pytest.fixture(scope="module")
def test_logit():
    from pycmtensor.expressions import Beta

    x = aet.vector("x")
    beta = aet.scalar("beta")

    with pytest.raises(NotImplementedError):
        functions.logit(utility=None)

    y = functions.logit(utility=beta * x).eval(
        {x: np.array([0.1, 0.4, 0.6, 0.2]), beta: np.array(1.0)}
    )

    assert y.shape[0] == 4
    assert np.allclose(y, np.array([0.19593432, 0.26448367, 0.32304109, 0.21654092]))
    assert y.sum() == 1.0

    x2 = aet.vector("x2")
    asc = aet.scalar("asc")

    y = functions.logit(utility=[beta * x, asc], avail=[np.ones(4), np.ones(4)]).eval(
        {x: np.array([0.1, 0.4, 0.6, 0.2]), beta: np.array(1.0), asc: np.array(0.07)}
    )

    assert len(y.shape) == 2
    assert y.shape[0] == 2
    assert y.shape[1] == 4
    assert np.allclose(y.sum(axis=0), np.array([1.0, 1.0, 1.0, 1.0]))

    asc_alt = Beta("asc_alt", value=0.07)
    y_2 = functions.logit(
        utility=[beta * x, asc_alt], avail=[np.ones(4), np.ones(4)]
    ).eval({x: np.array([0.1, 0.4, 0.6, 0.2]), beta: np.array(1.0)})

    assert np.allclose(y, y_2)

    with pytest.raises(ValueError):
        functions.logit(utility=[beta * x, asc_alt], avail=[np.ones(4)]).eval(
            {x: np.array([0.1, 0.4, 0.6, 0.2]), beta: np.array(1.0)}
        )
    return y_2


def test_log_likelihood(test_logit):
    prob = aet.as_tensor_variable(test_logit)
    y = aet.ivector("y")

    ll = functions.log_likelihood(prob, y).eval({y: np.array([0, 0, 0, 0])})
    ll = functions.log_likelihood(prob, y, index=np.arange(4)).eval(
        {y: np.array([0, 0, 0, 0])}
    )

    assert np.allclose(np.round(ll, 3), np.array(-2.313))


def test_rmse():
    rng = np.random.default_rng(123)
    y = aet.vector("y")
    y_hat = aet.vector("y_hat")

    r = functions.rmse(y_hat, y).eval(
        {y: np.array([0.1, 0.2, 0.3, 0.4]), y_hat: np.array([0.0, 0.0, 0.0, 0.0])}
    )

    assert np.round(r, 3) == 0.274

    with pytest.raises(ValueError):
        functions.rmse(y_hat, aet.matrix("y"))

    y_hat = aet.matrix("y_hat")
    y = aet.matrix("y")

    r = functions.rmse(y_hat, y).eval(
        {
            y: rng.normal(size=(3, 5)),
            y_hat: rng.normal(size=(3, 5)),
        }
    )

    assert np.round(r, 3) == 1.494


def test_mae():
    rng = np.random.default_rng(123)
    y = aet.vector("y")
    y_hat = aet.vector("y_hat")

    r = functions.mae(y_hat, y).eval(
        {y: np.array([0.1, 0.2, 0.3, 0.4]), y_hat: np.array([0.0, 0.0, 0.0, 0.0])}
    )

    assert np.round(r, 3) == 0.25

    with pytest.raises(ValueError):
        functions.mae(y_hat, aet.matrix("y"))

    y_hat = aet.matrix("y_hat")
    y = aet.matrix("y")

    r = functions.mae(y_hat, y).eval(
        {
            y: rng.normal(size=(3, 5)),
            y_hat: rng.normal(size=(3, 5)),
        }
    )

    assert np.round(r, 3) == 1.271


def test_kl_divergence():
    p = aet.vector("p")
    q = aet.vector("q")

    loss = aesara.function([p, q], functions.kl_divergence(p, q))
    assert round(float(loss([0.2, 0.1, 0.4, 0.3], [0.3, 0.1, 0.1, 0.5])), 4) == 0.3202
    assert float(loss([0.2, 0.1, 0.4, 0.3], [0.2, 0.1, 0.4, 0.3])) == float(0.0)


def test_kl_multivar_norm_univariate():
    m0 = aet.scalar("m0")
    v0 = aet.scalar("v0")
    m1 = aet.scalar("m1")
    v1 = aet.scalar("v1")

    kld = functions.kl_multivar_norm(m0, v0, m1, v1)
    loss = aesara.function([m0, v0, m1, v1], kld)
    assert m0.ndim == v0.ndim == 0
    assert m1.ndim == v1.ndim == 0

    assert round(float(loss(0, 1, 0, 1)), 2) == 0

    with pytest.raises(ValueError):
        kld = functions.kl_multivar_norm(m0, v0, m1, aet.vector())

    g = aet.grad(kld, v0, disconnected_inputs="ignore")
    grad = aesara.function([m0, v0, m1, v1], g)
    assert round(float(grad(3, 2, 0, 1)), 2) == 0.25

    g = aet.grad(kld, m0, disconnected_inputs="ignore")
    grad = aesara.function([m0, v0, m1, v1], g)
    assert round(float(grad(4, 2, 0, 1))) == 4


def test_kl_multivar_norm_1():
    rng = np.random.default_rng(42069)

    m0 = aet.vector("m0")
    v0 = aet.matrix("v0")
    m1 = aet.scalar("m1")
    v1 = aet.scalar("v1")

    kld = functions.kl_multivar_norm(m0, v0, m1, v1)
    loss = aesara.function([m0, v0, m1, v1], kld)

    output = loss([1, 2, 1], np.diag(rng.uniform(0, 1, 3)), 0, 1)
    assert round(float(output), 3) == 4.109

    g = aet.grad(kld, m0, disconnected_inputs="ignore")
    grad = aesara.function([m0, v0, m1, v1], g)

    output = grad([1, 2, 1], np.diag(rng.uniform(0, 1, 3)), 0, 1)
    assert [round(float(o), 2) for o in output] == [1, 2, 1]

    g = aet.grad(kld, v0, disconnected_inputs="ignore")
    grad = aesara.function([m0, v0, m1, v1], g)

    output = grad([1, 2, 1], np.diag(rng.uniform(0, 1, 3)), 0, 1)
    updates = [round(float(o), 2) for o in output.flatten()]
    assert len(updates) == 9

    assert updates[0] == -0.19


def test_kl_multivar_norm_2():
    rng = np.random.default_rng(42069)

    m0 = aet.vector("m0")
    v0 = aet.matrix("v0")
    m1 = aet.vector("m1")
    v1 = aet.matrix("v1")

    kld = functions.kl_multivar_norm(m0, v0, m1, v1)
    loss = aesara.function([m0, v0, m1, v1], kld)

    M0 = rng.uniform(0, 3, 3)  # ndim=1
    V0 = np.diag(rng.uniform(0, 1, 3))  # ndim=2
    M1 = rng.uniform(0, 3, 3)  # ndim=1
    V1 = np.diag(rng.uniform(0, 1, 3))  # ndim=2

    output = loss(M0, V0, M1, V1)
    assert round(float(output), 3) == 9.03

    # check anti-symmetry
    output = loss(M1, V1, M0, V0)
    assert round(float(output), 3) == 6.63


# class TestFunctions:
# @pytest.fixture
# def rng(self):
#     return np.random.default_rng(42069)

# @pytest.fixture
# def logit(self, utility, availability):
#     U = utility
#     AV = availability
#     return functions.logit(U, AV)

# @pytest.fixture
# def loglikelihood(self, logit, swissmetro_db):
#     prob = logit
#     db = swissmetro_db
#     return functions.log_likelihood(prob, db.y)

# def test_logit(self, utility, availability):
#     U = utility
#     AV = availability

#     assert len(U) == len(AV)
#     prob = functions.logit(U, AV)
#     assert prob.ndim == 2

#     with pytest.raises(ValueError):
#         functions.logit(U, AV[:-1])

#     U = aet.stack(U).flatten(2)
#     prob = functions.logit(U, AV)
#     assert prob.ndim == 2

#     prob = functions.logit(U)
#     assert type(prob) == TensorVariable

# def test_log_likelihood(self, loglikelihood):
#     assert loglikelihood.ndim == 0

# def test_errors(self, logit, swissmetro_db):
#     prob = logit
#     db = swissmetro_db
#     err = functions.errors(prob, db.y)
#     assert err.ndim == 0

#     with pytest.raises(NotImplementedError):
#         y_test = aet.vector("y")
#         functions.errors(prob, y_test)

# def test_hessians(self, mnl_model):
#     h = functions.hessians(mnl_model.cost, mnl_model.betas)
#     assert h.ndim == 2
#     assert list(h.shape.eval()) == [4, 4]

# def test_gradient_vector(self, mnl_model):
#     gradients = functions.gradient_vector(mnl_model.cost, mnl_model.betas)
#     assert gradients.ndim == 1
#     assert list(gradients.shape.eval()) == [4]
