from django.apps import AppConfig
from IntentClassification import Text_Classify_tfhub
from IntentClassification import Parameters
from IntentClassification.Parameters import logger


class IntentClassificationConfig(AppConfig):
    name = 'IntentClassification'
    estimatorObject = None    
       
    def ready(self):
      logger.info ('Intent App ready\n')

      try:
        logger.info ('Training Intents in app start up')
        #Text_Classify_tfhub.createTestData()
        #Text_Classify_tfhub.downLoad_NNLM_Model()
        Text_Classify_tfhub.trainModel ()
         
        IntentClassificationConfig.estimatorObject = Text_Classify_tfhub.make_estimator(Parameters.model_dir)

      except (RuntimeError, TypeError, NameError) as ex:
         logger.error ('Exception while training Intent app')
         logger.error (ex)
         pass

    
    def get_EstimatorObject():
        logger.debug ('get estimator object')
        
        if IntentClassificationConfig.estimatorObject is None:
            logger.info ('Estimator object not initialised, initialising')
            IntentClassificationConfig.estimatorObject = Text_Classify_tfhub.make_estimator(Parameters.model_dir)

        return IntentClassificationConfig.estimatorObject
