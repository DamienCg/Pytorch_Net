import pandas as pd
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from pretty_confusion_matrix import pp_matrix_from_data
import matplotlib.pyplot as plt

""" For umbalanced dataset """
def weight_class(dataset,train_subset):

    y_train_indices = train_subset.indices
    y_train = [dataset.y[i] for i in y_train_indices]

    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)

    return WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                 len(samples_weight))

def print_best_hyper_result(ListofResult):
    BestParam = max(ListofResult, key=lambda x: x['f1_score_macro'])
    print(BestParam["f1_score_macro"])
    print(BestParam["report"])
    # plot loss
    plt.plot(BestParam["loss"])
    plt.title("Number of epochs: {} - Hidden size: {} - Mini batch: {} - Learning rate: {} - Momentum: {}"
              .format(BestParam["num_epochs"], BestParam["hidden_size"],
                      BestParam["batch"],BestParam["learning_rate"],BestParam["momentum"]))
    plt.show()
    pp_matrix_from_data(BestParam["y_test"], BestParam["y_pred"], cmap=plt.cm.RdBu)


def save_result(ListofResult):
    for i in ListofResult:
        i.pop("report", None)
        i.pop("loss", None)
        i.pop("y_pred", None)
        i.pop("y_test", None)

    df = pd.DataFrame(ListofResult)
    print(df.sort_values(by= 'f1_score_macro', ascending=False))
    df.to_csv('BestParam.csv', index=False)



