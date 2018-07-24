# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import logging
from IntentClassification.Parameters import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from IntentClassification import Parameters
import json
from IntentClassification.Parameters import logger

# To create train test split and save to file
def createTestData():
   logger.info ('Starting training')
   xl = pd.ExcelFile( Parameters.data_Path + Parameters.dataFileName)

   Training_Matrix = []
   Training_Label = []
   Training_Label_Str = []
   
   logger.info ('Reading training data')
   
   for sheet in xl.sheet_names:
## Load a sheet into a DataFrame by name: df
     df = xl.parse(sheet)
     for value in df.values:
        Training_Matrix.append(str(value))
        Training_Label_Str.append(sheet)

   le = LabelEncoder()
   Training_Label = list(le.fit_transform(Training_Label_Str))
   logger.info (len (Training_Matrix))
   logger.info (len (Training_Label))

   x_train = np.array (Training_Matrix)
   y_train = np.array (Training_Label)
   y_train_str = np.array (Training_Label_Str)
   
   logger.info ('Split training data')
   sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
   for train_index, test_index in sss.split(x_train, y_train):
     logger.debug("TRAIN:", train_index, "TEST:", test_index)
     X_train_new, X_test_new = x_train[train_index], x_train[test_index]
     y_train_new, y_test_new = y_train[train_index], y_train[test_index]
     y_train_str, y_test_str = y_train_str[train_index], y_train_str[test_index]

   logger.info ('Save training data')
   df = pd.DataFrame(
        {Parameters.dataColumn: X_train_new,
     Parameters.labelColumn: y_train_new,
     Parameters.labelStrColumn: y_train_str
    })
        
   df.to_excel (Parameters.data_Path + Parameters.trainFileName)

   logger.info ('Save testing data')
   df = pd.DataFrame(
        {Parameters.dataColumn: X_test_new,
     Parameters.labelColumn: y_test_new,
     Parameters.labelStrColumn: y_test_str
    })
   df.to_excel (Parameters.data_Path + Parameters.testFileName)

   logger.info ('Completed create TestData')

# To download model
def downLoad_NNLM_Model():

    os.environ['http_proxy'] = Parameters.http_proxy 
    os.environ['HTTP_PROXY'] = Parameters.http_proxy
    os.environ['https_proxy'] = Parameters.https_proxy
    os.environ['HTTPS_PROXY'] = Parameters.https_proxy

    embed = hub.Module(Parameters.module_spec_Url)

# To get predictions
def get_predictions(estimator, input_fn):
  return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

# To train model
def trainModel (): 
#
# Reduce logging output.
  tf.logging.set_verbosity(tf.logging.DEBUG)

  log = logging.getLogger('tensorflow')
  log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
  fh = logging.FileHandler(Parameters.logpath + Parameters.tensorFlowLogFile)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  log.addHandler(fh)


  train_data =  pd.read_excel(Parameters.data_Path + Parameters.trainFileName)
  train_x = pd.DataFrame(train_data [Parameters.dataColumn])
  train_y  = pd.to_numeric(train_data [Parameters.labelColumn])
  

  sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
  for train_index, test_index in sss.split(train_x, train_y):
    log.debug("TRAIN:" + str(train_index) + "TEST:" + str( test_index))
    X_train_new, X_test_new = train_x.iloc[train_index], train_x.iloc[test_index]
    y_train_new, y_test_new = train_y.iloc[train_index], train_y.iloc[test_index]
  
  # Training input on the whole training set with no limit on training epochs.
  train_input_fn = tf.estimator.inputs.pandas_input_fn(
    X_train_new, y_train_new, num_epochs=None, shuffle=True)

# Prediction on the whole training set.
  eval_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    X_train_new, y_train_new, shuffle=False)
# Prediction on the test set.
  eval_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    X_test_new, y_test_new, shuffle=False)

  embedded_text_feature_column = hub.text_embedding_column(
    key=Parameters.dataColumn, 
    module_spec=Parameters.module_Spec_Path)


  estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=Parameters.n_classes,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003), model_dir=Parameters.model_dir)

# Training for 1,000 steps means 128,000 training examples with the default
# batch size. This is roughly equivalent to 5 epochs since the training dataset
# contains 25,000 examples.
  log.info ('Training chatbot')
  estimator.train(input_fn=train_input_fn, steps=3000);

  log.info ('Evaluating chatbot')

  evaluateModel (eval_train_input_fn, eval_test_input_fn, y_test_new, estimator);

  testModel (estimator)

  return estimator

# To evaluate model
def evaluateModel (eval_train_input_fn, eval_test_input_fn, y_test_new, estimator):
  train_eval_result = estimator.evaluate(input_fn=eval_train_input_fn)
  test_eval_result = estimator.evaluate(input_fn=eval_test_input_fn)

  logger.info( "Training set accuracy: {accuracy}".format(**train_eval_result))
  logger.info( "Test set accuracy: {accuracy}".format(**test_eval_result))

  with tf.Graph().as_default():
    
     pred = get_predictions(estimator, eval_test_input_fn)
     cm = tf.confusion_matrix(y_test_new, pred)
  
     with tf.Session() as session:
       cm_out = session.run(cm)

# Normalize the confusion matrix so that each row sums to 1.
       cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

       sns.heatmap(cm_out, annot=True, xticklabels=Parameters.LABELS, yticklabels=Parameters.LABELS);
       plt.xlabel("Predicted");
       plt.ylabel("True");

     logger.info ("\nPrecision:  ->" + str(precision_score(y_test_new, pred, average='micro')))
     logger.info ("\nRecall:  ->" + str(recall_score(y_test_new, pred, average='micro')))
     logger.info ("\nf1_score:  ->"+ str(f1_score(y_test_new, pred, average='micro')))


# To test model
def testModel (estimator):

   test_data =  pd.read_excel(Parameters.data_Path + Parameters.testFileName)
   test_x = pd.DataFrame(test_data [Parameters.dataColumn])
   test_y  = pd.DataFrame(test_data [Parameters.labelColumn])

# Prediction on the test set.
   predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_x, test_y, shuffle=False)


   test_pred_result = estimator.evaluate(input_fn=predict_test_input_fn)
   logger.info( "Test set accuracy: {accuracy}".format(**test_pred_result))

   test_pred_result = estimator.predict(input_fn=predict_test_input_fn)

   pred = []
   for result in test_pred_result:
      pred.append (result ['class_ids'])

   pred_y  = pd.DataFrame(pred)

   result_df = pd.concat([test_x, test_y, pred_y], axis=1, ignore_index=True)
   
   i = 0
   for index, row in result_df.iterrows():
    if int ( (row[1])) != int ( (row[2])) :
       logger.debug ( str(row[0]) + ' : ' + str(row[2]))
       i = i +1

   logger.info ('Incorrect predictions:' + str(i))

# To create estimator object
def make_estimator(model_dir):

    logger.info ('make_estimator')
    config = tf.estimator.RunConfig (model_dir=model_dir)
    embedded_text_feature_column = hub.text_embedding_column(
    key=Parameters.dataColumn, 
    module_spec=Parameters.module_Spec_Path)
    logger.info (embedded_text_feature_column)

    estimator = tf.estimator.DNNClassifier(config = config,
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=Parameters.n_classes,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))
        

    return estimator

# To predict user input class
def predictInput ( inputString):
   
   result = None
   from IntentClassification.apps import IntentClassificationConfig
   estimator = IntentClassificationConfig.get_EstimatorObject()
   
   predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    pd.DataFrame([inputString], columns = [Parameters.dataColumn]), shuffle=False)
   predictions = estimator.predict(input_fn=predict_test_input_fn)
   logger.info (predictions)
   for prediction in predictions:
    logger.info( prediction)
    logger.info (np.max(prediction[Parameters.prob]))
    result = json.dumps ({Parameters.Label : Parameters.LABELS[prediction [Parameters.class_id][0]], Parameters.probability: str(np.max(prediction[Parameters.prob]))})
    logger.info('Prediction:'+ result)

   return result;
