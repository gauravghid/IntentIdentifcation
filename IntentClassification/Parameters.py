
logpath = "."

app_url = "http://10.32.215.23:8000/Intent/"
data_Path = './Train_data/'
dataFileName = 'SourceData.xlsx'

trainFileName = 'Train_Intents.xlsx'
testFileName = 'Test_Intents.xlsx'

Label = 'Label'
class_id = 'class_ids'
probability = 'Probability'
tensorFlowLogFile = 'tensorflow.log'
dataColumn = 'sentence'
labelColumn = 'label'
labelStrColumn = 'labelStr'
module_spec_Url = "https://tfhub.dev/google/nnlm-en-dim128/1"
module_Spec_Path = './model/32f2b2259e1cc8ca58c876921748361283e73997'
n_classes = 5
model_dir = '/home/osboxes/Downloads/models/chatbot'
prob = 'probabilities'
default = 'default'

LABELS = [
    'ApprovalPolicy', 'InvoiceStatus', 'InvoiceSubProcess', 'Rushpayment','SmallTalk'
]

import logging

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

fileHandler = logging.FileHandler("{0}/{1}.log".format(".", "Intent"))
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

http_proxy = ''
https_proxy = ''

