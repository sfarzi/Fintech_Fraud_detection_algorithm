# -------------------------------------------------------------------
#                       Import Files and libraries
# -------------------------------------------------------------------
# IMPORTANT!    ====>   !First Import all libraries in 'Libraries.py
# NOTE!         ====>   !If your not using COLAB, You need to install
#                        (1)sklearn, (2)numpy, (3)pandas, (4)pyod,
#                        (5)cython, (6)matplotlib, (7)keras, (8)Tensorflow
#                        (9)XGBoost
#                        from File/settings/project/project interpreter
# -------------------------------------------------------------------
from __future__ import division
from __future__ import print_function
from Libraries import importLibs
importLibs()
import PrepareData
import MO_GAAL
import XGBOD

# -------------------------------------------------------------------
#             Read & Prepare data from Excel files
# -------------------------------------------------------------------
Type = "L"
X_train = PrepareData(Type)                                           # Type: L for Loan, D for Deposit,
                                                                      #       C for Cost, I for Income

# -------------------------------------------------------------------
#             Create an object from MO-GAAL class
# -------------------------------------------------------------------
FraudModel = MO_GAAL(k=7, stop_epochs=120, lr_d=0.01, lr_g=0.0001,    # stop_epochs : The number of epochs of training.
                     decay=1e-05, momentum=0.9, contamination=0.1)    # lr_d/g : float, optional (default=0.01/0.0001)
                                                                      #          The learn rate of the discriminator/generator.
                                                                      # decay : float, optional (default=1e-6)
                                                                      #         The decay parameter for SGD.
                                                                      # momentum : float, optional (default=0.9)
                                                                      #            The momentum parameter for SGD.
                                                                      # k : int, optional (default=10)
                                                                      #     The number of sub generators.

# -------------------------------------------------------------------
#       Now! Start generating outlier samples using MO-GAAL
# -------------------------------------------------------------------
FraudModel.fit(X_train)

# -------------------------------------------------------------------
#           Print generated samples and some information
# -------------------------------------------------------------------
print("Generated outliers:\n", FraudModel.generated_outliers, "\n")
print("Number of generated outliers: " , FraudModel.generated_outliers.shape[0])
print("Number of features in each generated outlier: " , FraudModel.generated_outliers.shape[1] , "\n")

# -------------------------------------------------------------------
#            Save the generated samples for further usage
# -------------------------------------------------------------------
with open('Loan_GeneratedOutliers.npy', 'wb') as f:
    np.save(f, FraudModel.generated_outliers)

# -------------------------------------------------------------------
#            Concat Real-Data with Generated outliers
# -------------------------------------------------------------------
New_X_train = np.concatenate((FraudModel.generated_outliers, X_train))
New_X_train_Y = np.array([1] * FraudModel.generated_outliers.shape[0] + [0] * X_train.shape[0])
All_Data = np.concatenate((New_X_train, New_X_train_Y[:,None]), axis=1)

# -------------------------------------------------------------------
#           Print some information about data
# -------------------------------------------------------------------
print("New_X_train: \n" , New_X_train, "\n")
print("New_X_train_Y: \n", New_X_train_Y , "\n")
print("All Data: \n", All_Data, "\n")

# -------------------------------------------------------------------
#             Create an object from XGBOD class
# -------------------------------------------------------------------
FraudModel_2 = XGBOD(estimator_list=None, standardization_flag_list=None,
                     max_depth=3, learning_rate=0.2, n_estimators=100,
                     silent=True, objective='binary:logistic', booster='gbtree',
                     n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
                     max_delta_step=0, subsample=1, colsample_bytree=1,
                     colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                     scale_pos_weight=1, base_score=0.5, random_state=0)

# -------------------------------------------------------------------
#             Prepare data for K-Fold cross validation
# -------------------------------------------------------------------
k = 5                                                                 # number of folds
np.random.shuffle(All_Data)
X = All_Data[:, :-1]
Y = All_Data[:, -1]
rng = math.ceil(All_Data.shape[0] / k)                                # number of samples in each fold

# -------------------------------------------------------------------
#           Train XGBOD on 4 Fold and Test it on 1 Fold
# -------------------------------------------------------------------
for i in range(k):
    print("\n", "*" * 40, " Fold ", i, " ", "*" * 40)

    # +++++++++++++++++++ Hold out 1 Fold and Use 4 Fold ++++++++++++++++++++
    # ==== Train ====
    TrX = np.concatenate((X[:(i * rng), :], X[((i + 1) * rng):, :]))
    TrY = np.concatenate((Y[:(i * rng)], Y[((i + 1) * rng):]))
    print("Train X shape: ", TrX.shape)
    print("Train Y shape: ", TrY.shape)
    print("Train labels:", TrY, "\n")

    # ==== Test ====
    TeX = X[(i * rng):((i + 1) * rng), :]
    TeY = Y[(i * rng):((i + 1) * rng)]
    print("Test X shape: ", TeX.shape)
    print("Test Y shape: ", TeY.shape)
    print("Test labels:", TeY)

    TestIndxes = np.arange((i * rng), ((i + 1) * rng))
    print("Test Samples: ", TestIndxes, "\n")

    # ++++++++++++++++++++ Train Model on Training data +++++++++++++++++++++
    FraudModel_2.fit_predict_score(TrX, TrY)
    print("-" * 30, "Training Results", "-" * 30)
    Train_Y_pred = FraudModel_2.predict(TrX)                          # binary labels (X_test), y_test_pred
    Train_Y_scores = FraudModel_2.decision_function(TrX)              # raw outlier scores (X_test), y_test_scores
    Train_Y_proba = FraudModel_2.predict_proba(TrX)                   # outlier probability (X_test), y_test_proba
    print(classification_report(TrY, Train_Y_pred))

    # +++++++++++++++++++++++ Test Model on Test data +++++++++++++++++++++++
    print("-" * 30, "Testing Results", "-" * 30)
    Test_Y_pred = FraudModel_2.predict(TeX)                           # binary labels (X_test), y_test_pred
    Test_Y_scores = FraudModel_2.decision_function(TeX)               # raw outlier scores (X_test), y_test_scores
    Test_Y_proba = FraudModel_2.predict_proba(TeX)                    # outlier probability (X_test), y_test_proba
    print(classification_report(TeY, Test_Y_pred))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #    Visualize real an predicted labels (on a 2-dimensional subspace)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dim1 = 0
    dim2 = 1
    visualize("XGBOD_Fold" + str(i), TrX[:, [dim1, dim2]], TrY, TeX[:, [dim1, dim2]],
              TeY, Train_Y_pred, Test_Y_pred, show_figure=True, save_figure=True)