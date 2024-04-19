from django.db import models

# Create your models here.
# models.py
from django.db import models

class files_mod(models.Model):
    images = models.FileField(upload_to='files/')
# Create your models here.
from django.db import models
from django.contrib import auth

# Create your models here.
class User(auth.models.User,auth.models.PermissionsMixin):
    def __str__(self):
        return "@{}".format(self.username)