import os
from train_test import train_test
from submission import gen_submission

if os.path.exists('model.pt') == False:
    train_test()

gen_submission()