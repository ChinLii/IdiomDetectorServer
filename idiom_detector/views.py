import os
from audioop import reverse

from django.core.checks import messages
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader

from .forms import UploadFileForm
from django.shortcuts import render
from pip.utils import logging

from idiom_detector.VNAAD_extraction_SVM import Detectidioms
from .functions import handle_uploaded_file


# Create your views here

def index(request):
    return render(request, "index.html")


def training(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            print("pass")
            percentage = handle_uploaded_file(request.FILES['file'])
            request.session['data'] = percentage
            return HttpResponseRedirect('/successful')

    else:
        form = UploadFileForm()
        print("Fail")
    return render(request, 'trainDataset.html', {'form': form})


def successful(request):
    data = request.session['data']
    data = data * 100
    return render(request, "successful.html", {'data': data})
