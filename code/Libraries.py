# ATTENTION!
# ===========================================================
# This file contains the list of all required libraries
# You should install/import them into your platform
# -----------------------------------------------------------
def importLibs():
    # -----------------------------------------------------------
    #                   1. Primary libraries
    # -----------------------------------------------------------
    import os
    import sys
    import numpy as np
    import scipy.sparse.csgraph
    import sklearn
    from sklearn import metrics
    from ast import literal_eval
    import pandas as pd
    import matplotlib.pyplot as plt
    import math
    from math import ceil
    from sklearn.metrics import classification_report
    from sklearn.metrics import precision_score, recall_score, accuracy_score

    # -----------------------------------------------------------
    #                       2. PyOD Libraries
    # -----------------------------------------------------------
    # pip install pyod                    # normal install
    # pip install --upgrade pyod          # or update if needed

    # -----------------------------------------------------------
    #                     3. MO-GAAL Libraries
    # -----------------------------------------------------------
    from collections import defaultdict
    from tempfile import TemporaryFile
    from sklearn.utils import check_array
    from sklearn.utils.validation import check_is_fitted
    from pyod.models.base import BaseDetector
    from pyod.models.gaal_base import create_discriminator
    from pyod.models.gaal_base import create_generator
    from pyod.models.base_dl import _get_tensorflow_version

    if _get_tensorflow_version() == 1:
        from keras.layers import Input
        from keras.models import Model
        from keras.optimizers import SGD
    else:
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import SGD

    # -----------------------------------------------------------
    #                   4. XGBOD Libraries
    # -----------------------------------------------------------
    # pip install cython

    from sklearn.metrics import roc_auc_score
    from sklearn.utils import check_array
    from sklearn.utils.validation import check_is_fitted
    from sklearn.utils.validation import check_X_y

    try:
        import xgboost
    except ImportError:
        print('please install xgboost first for running XGBOD')

    from xgboost.sklearn import XGBClassifier
    from pyod.models.base import BaseDetector
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.hbos import HBOS
    from pyod.models.ocsvm import OCSVM
    from pyod.models.loda import LODA
    # from pyod.models.iforest import IForest

    from pyod.utils.utility import check_parameter
    from pyod.utils.utility import check_detector
    from pyod.utils.utility import standardizer
    from pyod.utils.utility import precision_n_scores
    from pyod.utils.data import evaluate_print
    from pyod.utils.example import visualize
