# Created by Baole Fang at 10/31/23
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
import numpy as np
import pickle
import xgboost
import pandas as pd
import matplotlib.pyplot as plt


def load(path: str, idx: int = 0):
    data = np.load(path)
    X = []
    Y = []
    for x, y in zip(data['X'], data['Y']):
        ys = y.split(';')
        if idx < len(ys):
            X.append(x)
            Y.append(ys[idx])
    return X, np.array(Y)


def main(args):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    X, y = load(args.input, args.column)
    X_train, X_test, y_train, y_test = train_test_split(X[:10000], y[:10000], test_size=args.test, random_state=args.seed)
    model = xgboost.XGBClassifier(n_jobs=os.cpu_count())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred).astype(int)
    report = classification_report(y_test, y_pred)
    path = os.path.join(args.output, os.path.basename(args.input).split('.')[0], args.model)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(path, 'metrics.txt'), 'w') as f:
        f.write(report)
    df = pd.DataFrame(cm)
    df.to_csv(os.path.join(path, 'confusion.csv'), index=False, header=False)

    fig,ax=plt.subplots(figsize=(len(model.classes_)//4,len(model.classes_)//4))
    disp=ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax)
    plt.savefig(os.path.join(path, 'confusion.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train the classification problem')
    parser.add_argument('-i', '--input', help='path of the data', type=str, required=True)
    parser.add_argument('-c', '--column', help='column/index of label', type=int, default=0)
    parser.add_argument('-t', '--test', help='test set size', type=float, default=0.2)
    parser.add_argument('-o', '--output', help='path of output model', type=str, default='outputs')
    parser.add_argument('-m', '--model', help='model name', type=str, default='xgboost')
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
