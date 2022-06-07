import os.path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import git

# Get the astronet source code
git clone https://github.com/zdebeurs/exoplanet-ml.git

import sys
sys.path.append("exoplanet-ml/exoplanet-ml/")
from astronet.ops import training
from tf_util import config_util
from tf_util import configdict
from tf_util import estimator_runner
from rv_net import data, rv_model, estimator_util