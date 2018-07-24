from django.shortcuts import render

from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from IntentClassification import Text_Classify_tfhub

from IntentClassification.Parameters import logger


@csrf_exempt
def getIntent(request, inputString):
    
    logger.debug ('Request reached in Intent views')
    logger.info ('User input -->'+ inputString )
    response = Text_Classify_tfhub.predictInput(inputString)
    logger.debug ('response -->'+ response )

    return HttpResponse( response)
