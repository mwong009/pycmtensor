import dill as pickle
import pandas as pd

import pycmtensor as cmt
from pycmtensor.expressions import Beta
from pycmtensor.models import MNLogit
from pycmtensor.optimizers import Adam
from pycmtensor.results import Results

swissmetro = pd.read_csv("data/swissmetro.dat", sep="\t")
db = cmt.Database(name="swissmetro", pandasDatabase=swissmetro, choiceVar="CHOICE")
globals().update(db.variables)
# Removing some observations
db.data.drop(db.data[db.data["CHOICE"] == 0].index, inplace=True)
db.data["CHOICE"] -= 1  # set the first choice index to 0
db.choices = [0, 1, 2]
db.autoscale(
    variables=["TRAIN_CO", "TRAIN_TT", "CAR_CO", "CAR_TT", "SM_CO", "SM_TT"],
    default=100.0,
    verbose=False,
)

b_cost = Beta("b_cost", 0.0, None, None, 0)
b_time = Beta("b_time", 0.0, None, None, 0)
asc_train = Beta("asc_train", 0.0, None, None, 0)
asc_car = Beta("asc_car", 0.0, None, None, 0)
asc_sm = Beta("asc_sm", 0.0, None, None, 1)

U_1 = b_cost * db["TRAIN_CO"] + b_time * db["TRAIN_TT"] + asc_train
U_2 = b_cost * db["SM_CO"] + b_time * db["SM_TT"] + asc_sm
U_3 = b_cost * db["CAR_CO"] + b_time * db["CAR_TT"] + asc_car

# specify the utility function and the availability conditions
U = [U_1, U_2, U_3]
AV = [db["TRAIN_AV"], db["SM_AV"], db["CAR_AV"]]

mymodel = MNLogit(u=U, av=AV, database=db, name="Multinomial Logit")
mymodel.add_params(locals())

model = cmt.train(
    model=mymodel,
    database=db,
    optimizer=Adam,
)

with open("model.pkl", "wb") as f:
    model.export_to_pickle(f)


result = Results(model, db)
