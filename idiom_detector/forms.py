from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField()

class InputForm(forms.Form):
    text = forms.TextInput()
