# test_functions.py
import aesara.tensor as aet
import pytest
from aesara.tensor.var import TensorVariable

import pycmtensor.functions as functions


class TestFunctions:
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

    def test_errors(self, logit, swissmetro_db):
        prob = logit
        db = swissmetro_db
        err = functions.errors(prob, db.y)
        assert err.ndim == 0

        with pytest.raises(NotImplementedError):
            y_test = aet.vector("y")
            functions.errors(prob, y_test)

        with pytest.raises(ValueError):
            y_test = aet.imatrix("y")
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
