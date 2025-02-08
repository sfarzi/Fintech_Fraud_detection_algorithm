# ATTENTION!
# ===========================================================
# You have to change the File_Address based on the location
# that you've saved your data files (in xlsx format (Excel)).
# For example:
# DEPOSIT_ADDRESS = /content/drive/MyDrive/PROJECT_FOLDER/
#                   DATA_FOLDER/FileName.xlsx
# The Type parameter should be 'L' for 'Loan', 'C' for 'Cost',
# 'D' for 'Deposit', and 'I' for 'Income'
# -----------------------------------------------------------
def PrepareData(Type):
  File_Address = {"L": "LOAN_ADDRESS",
                  "C": "COST_ADDRESS",
                  "I": "INCOME_ADDRESS",
                  "D": "DEPOSIT_ADDRESS"
                  }
  if (Type != "A"):
    if (Type == "D"):
      data = pd.read_excel(File_Address[Type])
    elif (Type == "I"):
      data = pd.read_excel(File_Address[Type])
    elif (Type == "C"):
      data = pd.read_excel(File_Address[Type])
    elif (Type == "L"):
      data = pd.read_excel(File_Address[Type])
    training_data = data
    display(training_data)
    training_data.drop(training_data.columns[[0, 1]], axis=1, inplace=True)         # delete Bank Name and Year from data
  else:
    data1 = pd.read_excel(File_Address["L"])
    data2 = pd.read_excel(File_Address["I"])
    data3 = pd.read_excel(File_Address["D"])
    data4 = pd.read_excel(File_Address["C"])
    data1.drop(data1.columns[[0, 1]], axis=1, inplace=True)
    data2.drop(data2.columns[[0, 1]], axis=1, inplace=True)
    data3.drop(data3.columns[[0, 1]], axis=1, inplace=True)
    data4.drop(data4.columns[[0, 1]], axis=1, inplace=True)
    training_data = np.concatenate((data1, data2), axis=1)
    training_data = np.concatenate((training_data, data3), axis=1)
    training_data = np.concatenate((training_data, data4), axis=1)

  display(training_data)

  training_data.replace({'/':'.'}, regex=True,  inplace=True)                     # replace / characters with . to further convert string numbers to float
  X_train = (training_data.astype(float)).to_numpy()                              # convert to float, then convert to numpy array

  print(X_train)
  print("\nNumber of Samples:" + str(X_train.shape[0]))
  print("Number of Features:" + str(X_train.shape[1]))

  return X_train