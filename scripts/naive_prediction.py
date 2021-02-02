from image_data_module import TrainTestData
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score
from collections import Counter

data_loader = TrainTestData()


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# salience
mae_all_random = []
mape_all_random = []
mae_all_naive = []
mape_all_naive = []
for cv_id in range(0, 5):
    (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_salience(cv_name=str(cv_id))
    y_pred_random = np.random.normal(np.mean(Y_train), np.std(Y_train), Y_test.shape)
    y_pred_naive = np.full((Y_test.shape), np.mean(Y_train))
    mae_random = mean_absolute_error(Y_test, y_pred_random)
    mape_random = mean_absolute_percentage_error(Y_test, y_pred_random)
    mae_all_random.append(mae_random)
    mape_all_random.append(mape_random)
    mae_naive = mean_absolute_error(Y_test, y_pred_naive)
    mape_naive = mean_absolute_percentage_error(Y_test, y_pred_naive)
    mae_all_naive.append(mae_naive)
    mape_all_naive.append(mape_naive)
print("MAEs random:", mae_all_random, "AVG MAE random:", np.mean(np.asarray(mae_all_random)))
print("MAPEs random:", mape_all_random, "AVG MAPE random:", np.mean(np.asarray(mape_all_random)))
print("MAEs naive:", mae_all_naive, "AVG MAE naive:", np.mean(np.asarray(mae_all_naive)))
print("MAPEs naive:", mape_all_naive, "AVG MAPE naive:", np.mean(np.asarray(mape_all_naive)))

# classification
acc_random = []
acc_naive = []
for cv_id in range(0, 5):
    (X_train, Y_train), (X_test, Y_test) = data_loader.get_train_test_binary(cv_name=str(cv_id))
    y_pred_random = np.random.randint(2, size=Y_test.shape[0])
    y_pred_naive = np.full((Y_test.shape), list(Counter(Y_train).keys())[0])
    acc_random.append(accuracy_score(Y_test, y_pred_random))
    acc_naive.append(accuracy_score(Y_test, y_pred_naive))
print("ACCs random:", acc_random, "AVG ACC random:", np.mean(np.asarray(acc_random)))
print("ACCs naive:", acc_naive, "AVG ACC naive:", np.mean(np.asarray(acc_naive)))
