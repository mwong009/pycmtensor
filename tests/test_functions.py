# test_functions.py
import aesara
import aesara.tensor as aet
import numpy as np
import pytest
from aesara.tensor.var import TensorVariable

import pycmtensor.functions as functions


class TestFunctions:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42069)

    @pytest.fixture
    def logit(self, utility, availability):
        U = utility
        AV = availability
        return functions.logit(U, AV)

    @pytest.fixture
    def loglikelihood(self, logit, swissmetro_db):
        prob = logit
        db = swissmetro_db
        return functions.log_likelihood(prob, db.y)

    def test_logit(self, utility, availability):
        U = utility
        AV = availability

        assert len(U) == len(AV)
        prob = functions.logit(U, AV)
        assert prob.ndim == 2

        with pytest.raises(ValueError):
            functions.logit(U, AV[:-1])

        U = aet.stack(U).flatten(2)
        prob = functions.logit(U, AV)
        assert prob.ndim == 2

        prob = functions.logit(U)
        assert type(prob) == TensorVariable

    def test_log_likelihood(self, loglikelihood):
        assert loglikelihood.ndim == 0

    def test_kl_divergence(self):
        p = aet.vector("p")
        q = aet.vector("q")

        loss = aesara.function([p, q], functions.kl_divergence(p, q))
        assert (
            round(float(loss([0.2, 0.1, 0.4, 0.3], [0.3, 0.1, 0.1, 0.5])), 4) == 0.3202
        )
        assert float(loss([0.2, 0.1, 0.4, 0.3], [0.2, 0.1, 0.4, 0.3])) == float(0.0)

    def test_kl_multivar_norm_univariate(self):
        m0 = aet.scalar("m0")
        v0 = aet.scalar("v0")
        m1 = aet.scalar("m1")
        v1 = aet.scalar("v1")

        kld = functions.kl_multivar_norm(m0, v0, m1, v1)
        loss = aesara.function([m0, v0, m1, v1], kld)
        assert loss(0, 1, 0, 1) == 0

        with pytest.raises(ValueError):
            kld = functions.kl_multivar_norm(m0, v0, m1, aet.vector())

        g = aet.grad(kld, v0, disconnected_inputs="ignore")
        grad = aesara.function([m0, v0, m1, v1], g)
        assert float(grad(3, 2, 0, 1)) == 0.25

        g = aet.grad(kld, m0, disconnected_inputs="ignore")
        grad = aesara.function([m0, v0, m1, v1], g)
        assert float(grad(4, 2, 0, 1)) == 4

    def test_kl_multivar_norm_1(self, rng):
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
        assert [float(o) for o in output] == [1, 2, 1]

        g = aet.grad(kld, v0, disconnected_inputs="ignore")
        grad = aesara.function([m0, v0, m1, v1], g)
        output = grad([1, 2, 1], np.diag(rng.uniform(0, 1, 3)), 0, 1)
        updates = [round(float(o), 2) for o in output.flatten()]
        assert len(updates) == 9
        assert updates[0] == -0.19

    def test_kl_multivar_norm_2(self, rng):
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

    def test_errors(self, logit, swissmetro_db):
        prob = logit
        db = swissmetro_db
        err = functions.errors(prob, db.y)
        assert err.ndim == 0

        with pytest.raises(NotImplementedError):
            y_test = aet.vector("y")
            functions.errors(prob, y_test)

    def test_hessians(self, mnl_model):
        h = functions.hessians(mnl_model.ll, mnl_model.betas)
        assert h.ndim == 2
        assert list(h.shape.eval()) == [4, 4]

    def test_bhhh(self, swissmetro_db, mnl_model):
        db = swissmetro_db
        data = db.pandas.inputs(mnl_model.inputs)
        bhhh = mnl_model.BHHH(*data)
        assert bhhh.ndim == 2
        assert list(bhhh.shape) == [4, 4]

    def test_gnorm(self, mnl_model):
        gn = functions.gnorm(mnl_model.cost, mnl_model.betas)
        assert gn.ndim == 0
