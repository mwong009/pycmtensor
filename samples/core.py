import numpy as np
import pandas as pd

import pycmtensor as cmt
from pycmtensor.expressions import Beta
from pycmtensor.models import MNL
from pycmtensor.scheduler import StepLR
from pycmtensor.statistics import elasticities

cmt.logger.set_level(cmt.logger.INFO)

swissmetro = pd.read_csv("../data/swissmetro.dat", sep="\t")
swissmetro.drop(swissmetro[swissmetro["CHOICE"] == 0].index, inplace=True)
swissmetro["CHOICE"] -= 1  # set the first choice index to 0
db = cmt.Data(df=swissmetro, choice="CHOICE")
db.autoscale_data(except_for=["ID", "ORIGIN", "DEST"])  # scales dataset
db.split_db(split_frac=0.8)  # split dataset

b_cost = Beta("b_cost", 0.0, None, None, 0)
b_time = Beta("b_time", 0.0, None, None, 0)
asc_train = Beta("asc_train", 0.0, None, None, 0)
asc_car = Beta("asc_car", 0.0, None, None, 0)
asc_sm = Beta("asc_sm", 0.0, None, None, 1)

U_1 = b_cost * db["TRAIN_CO"] + b_time * db["TRAIN_TT"] + asc_train
U_2 = b_cost * db["SM_CO"] + b_time * db["SM_TT"] + asc_sm
U_3 = b_cost * db["CAR_CO"] + b_time * db["CAR_TT"] + asc_car

# specify the utility function and the availability conditions
U = [U_1, U_2, U_3]  # utility
AV = [db["TRAIN_AV"], db["SM_AV"], db["CAR_AV"]]  # availability

mymodel = MNL(U, AV, locals(), db, name="MNL")
mymodel.config.set_hyperparameter("max_steps", 200)
mymodel.config.set_lr_scheduler(StepLR())
mymodel.train(db)

print(elasticities(mymodel, db, 0, "TRAIN_TT"))
print(mymodel.results.beta_statistics())
print(mymodel.results.model_statistics())
print(mymodel.results.model_correlation_matrix())
print(mymodel.results.benchmark())

# predictions
print(mymodel.predict(db, return_choices=False))
print(np.unique(mymodel.predict(db), return_counts=True))
