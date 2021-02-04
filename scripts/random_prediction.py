from image_data_module import TrainTestData
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error
from collections import Counter

data_loader = TrainTestData()

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# salience
mse_all_random = []
mae_all_random = []
mape_all_random = []
for cv_id in range(0, 5):
    (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_salience(cv_name=str(cv_id))
    Y_train = Y_train / 5.0
    Y_test = Y_test / 5.0
    y_pred_random = np.random.normal(np.mean(Y_train), np.std(Y_train), Y_test.shape)
    mse_random = mean_squared_error(Y_test, y_pred_random)
    mae_random = mean_absolute_error(Y_test, y_pred_random)
    mape_random = mean_absolute_percentage_error(Y_test, y_pred_random)
    mse_all_random.append(mse_random)
    mae_all_random.append(mae_random)
    mape_all_random.append(mape_random)
print("MSEs random:", mse_all_random, "AVG MSE random:", np.mean(np.asarray(mse_all_random)))
print("MAEs random:", mae_all_random, "AVG MAE random:", np.mean(np.asarray(mae_all_random)))
print("MAPEs random:", mape_all_random, "AVG MAPE random:", np.mean(np.asarray(mape_all_random)))

# classification
acc_random = []
for cv_id in range(0, 5):
    (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_binary(cv_name=str(cv_id))
    y_pred_random = np.random.randint(2, size=Y_test.shape[0])
    acc_random.append(accuracy_score(Y_test, y_pred_random))
print("ACCs random:", acc_random, "AVG ACC random:", np.mean(np.asarray(acc_random)))
