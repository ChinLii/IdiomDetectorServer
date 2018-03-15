from django.db import models


# Create your models here.
class Note(models.Model):
    idiom = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    # This method defines what we get when we ask for a particular instance of a model.

    def __str__(self):
        return '%s %s' % (self.idiom, self.created_at)
