import os
from audioop import reverse

from django.core.checks import messages
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader

from .forms import UploadFileForm
from .forms import InputForm
from django.shortcuts import render
from pip.utils import logging

from idiom_detector.VNAAD_extraction_SVM import Detectidioms
from .functions import handle_uploaded_file
from .functions import handle_input_sentence


# Create your views here

def index(request):
    if request.method == 'POST':
        form = InputForm(request.POST)
        if form.is_valid():
            sentence = request.POST.get("text", "")
            result = handle_input_sentence(sentence)
            request.session['result'] = result[0]
            return HttpResponseRedirect('/result')

    else:
        form = InputForm()

    return render(request, "index.html")


def training(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            percentage = handle_uploaded_file(request.FILES['file'])
            request.session['data'] = percentage
            return HttpResponseRedirect('/successful')

    else:
        form = UploadFileForm()
    return render(request, 'trainDataset.html', {'form': form})


def successful(request):
    data = request.session['data']
    data = data * 100
    return render(request, "successful.html", {'data': data})


def result(request):
    shortresult = request.session['result']
    if shortresult == 'I':
        result = 'Idiom'
    elif shortresult == 'L':
        result = 'Literal'
    elif shortresult == 'Q':
        result = 'Unknown'

    return render(request, "result.html", {'data': result})
