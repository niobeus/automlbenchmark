import logging
import os
import tempfile as tmp

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from frameworks.shared.callee import call_run, result, save_metadata, utils

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import roc_auc_score

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np

log = logging.getLogger(os.path.basename(__file__))

class DataSet:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __len__(self):
    return len(self.x)

  def __getitem__(self, i):
    return torch.tensor(self.x[i]), torch.tensor(self.y[i])

class Sampler:
  def __init__(self, X, y):
    self.dataset = DataSet(X, y)
  def sample(self, batch_size):
    n = len(self.dataset)
    idxs = torch.randperm(n)
    for i in range(0, n, batch_size):
      yield self.dataset[idxs[i: i + batch_size]]

def layer(input, output):
  return nn.Sequential(nn.Linear(input, output), nn.SELU(), nn.AlphaDropout(p=0.1))

class MLP(nn.Module):
  def __init__(self, n_input, n_hidden, n_output):
    super(MLP, self).__init__()
    self.n_output = n_output
    n_hidden = [n_input] + n_hidden
    layers = [layer(n_hidden[i], n_hidden[i+1]) for i in range(len(n_hidden) - 1)]
    layers.append(nn.Sequential(nn.Linear(n_hidden[-1], n_output)))
    
    self.model = nn.Sequential(*layers)
    
  def forward(self, x):
    x = self.model(x)
    return x

class NNClassifier:
  def __init__(self, MLP, batch_size, optimizer, lr, loss, device, max_epochs):
    self.MLP = MLP.to(device)
    self.sampler = Sampler
    self.batch_size = batch_size
    self.optimizer = optimizer(self.MLP.parameters(), lr=lr)
    self.lr = lr
    self.loss = loss
    self.device = device
    self.max_epochs = max_epochs
  
  def fit(self, X_train, y_train):
    self.MLP.train()
    for i in range(self.max_epochs):
      for batch in self.sampler(X_train, y_train).sample(self.batch_size):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        y_pred = self.MLP.forward(x)

        # print(y_pred, y.long())
        if self.MLP.n_output == 1:
          # y = y.unsqueeze(1)
          loss = self.loss(y_pred, y)
        else:
          loss = self.loss(y_pred, torch.max(y.long(), 1)[0])
        
        loss.backward()
        self.optimizer.step()
      # print('Epoch {}: train loss: {:.5f}'.format(i, loss.item()))
      # print(y_pred, torch.max(y,1)[0])
    # print(torch.max(y.long(),1)[0], y.long())
      
  def predict(self, X):
    self.MLP.eval()
    with torch.no_grad():
      if self.MLP.n_output == 1:
        return torch.round(torch.sigmoid(self.MLP.forward(X)))
      else:
        return torch.argmax(self.MLP.forward(X))
    
  def predict_proba(self, X):
    self.MLP.eval()
    with torch.no_grad():
      if self.MLP.n_output == 1:
        return torch.sigmoid(self.MLP.forward(X))
      else:
        return nn.functional.softmax(self.MLP.forward(X))

  def score(self, X_test, y_test, metric='acc'):
    self.MLP.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for batch in self.sampler(X_test, y_test).sample(self.batch_size):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)

        y_pred = self.MLP.forward(x)
        test_loss += self.loss(y_pred, y.long()).item()  # sum up batch loss
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(y.view_as(pred)).sum().item()

    if metric == 'acc':
      return 100. * correct / len(y_test)
    if metric == 'loss':
      return test_loss * self.batch_size / len(y_test)

def run(dataset, config):

    is_classification = config.type == 'classification'

    X_train, X_test = dataset.train.X_enc, dataset.test.X_enc
    y_train, y_test = dataset.train.y_enc, dataset.test.y_enc

    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')

    log.info("Running SNN")
    log.warning("We completely ignore the requirement to stay within the time limit.")
    log.warning("We completely ignore the advice to optimize towards metric: {}.".format(config.metric))
    
    estimator = NNClassifier if is_classification else None
    (_, y_train_counts) = np.unique(y_train, return_counts=True)
    n_input = len(X_train[0])
    n_hidden = len(X_train[0])
    n_hidden = [n_input, n_input, n_input // 2]
    n_output = len(y_train_counts)

    if n_output > 2:
      loss = nn.CrossEntropyLoss()
    else:
      loss = nn.BCEWithLogitsLoss()
      n_output = 1
    # loss = nn.CrossEntropyLoss()

    mlp = MLP(n_input=n_input, 
              n_hidden=n_hidden, 
              n_output=n_output)

    batch_size = 10
    epochs = 30
    
    net = estimator(mlp, 
                  batch_size=batch_size, 
                  optimizer=torch.optim.SGD, 
                  lr=0.01, 
                  loss= loss,
                  device='cuda:0', 
                  max_epochs=epochs)

    with utils.Timer() as training:
        net.fit(X_train, y_train)

    with utils.Timer() as predict:
      predictions = []
      idx = 0
      count = 0
      for x in X_test:
        x = torch.Tensor(x).to('cuda:0')
        pred_x = net.predict(x).to('cpu').tolist()
        predictions.append(pred_x)
        
        # if int(y_test[idx]) == int(pred_x):
        #   count += 1 
        # idx += 1

    xx_test = torch.Tensor(X_test).to('cuda:0')
    probabilities = net.predict_proba(xx_test).detach().to('cpu') if is_classification else None
    probabilities = probabilities.tolist()
        
    # print('TEST ACCURACY', count/len(y_test))
    if n_output==1:
      auc = roc_auc_score(y_test, predictions)
      print('TEST AUC ', auc)
    else:
      res = []
      [res.extend(l) for l in y_test]
      res = list(map(int, res))
      # print(res)
      # print(predictions)
      auc_ovo = roc_auc_score(res, probabilities, multi_class='ovo')
      auc_ovr = roc_auc_score(res, probabilities, multi_class='ovr')
      print('TEST AUC OVO {} OVR {}'.format(auc_ovo, auc_ovr))
    
    print('TEST LOSS (CrossEntropy) ', net.score(X_test, y_test))
    print('TEST ACCURACY ', net.score(X_test, y_test))
    print('TRAIN AND TEST FINISHED')

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  training_duration=training.duration,
                  predict_duration=predict.duration)


if __name__ == '__main__':
    call_run(run)
