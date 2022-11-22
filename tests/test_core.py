import pytest

from pycmtensor.expressions import Beta
from pycmtensor.functions import logit
from pycmtensor.models.MNL import MNL


class TestCore:
    @pytest.fixture
    def ascs(self):
        asc_train = Beta("asc_train", 0.0, None, None, 0)
        asc_sm = Beta("asc_sm", 0.0, None, None, 0)
        asc_car = Beta("asc_car", 0.0, None, None, 1)
        return {"asc_train": asc_train, "asc_sm": asc_sm, "asc_car": asc_car}

    @pytest.fixture
    def constants_only_U(self, ascs):
        globals().update(ascs)
        U_1 = asc_train
        U_2 = asc_sm
        U_3 = asc_car
        U = [U_1, U_2, U_3]
        return U

    def test_constants_only(self, ascs, swissmetro_db, constants_only_U, availability):
        db = swissmetro_db
        U = constants_only_U
        AV = availability

        assert len(U) == len(AV)
        prob = logit(U, AV)
        assert prob.ndim == 2

        model = MNL(db=db, params=ascs, utility=U, av=AV)
        model.train(db, max_steps=10)
        assert len(model.results.beta_statistics()) == 3
