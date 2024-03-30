from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address = models.CharField(max_length=300)
    gender = models.CharField(max_length=30)

class air_quality_type(models.Model):

    aid = models.CharField(max_length=30000)
    City=models.CharField(max_length=300)
    Date=models.CharField(max_length=300)
    PM2andhalf=models.CharField(max_length=300)
    PM10=models.CharField(max_length=300)
    NO=models.CharField(max_length=300)
    NO2=models.CharField(max_length=300)
    Nox=models.CharField(max_length=300)
    NH3=models.CharField(max_length=300)
    CO=models.CharField(max_length=300)
    SO2=models.CharField(max_length=300)
    O3=models.CharField(max_length=300)
    Benzene=models.CharField(max_length=300)
    Toluene=models.CharField(max_length=300)
    Xylene=models.CharField(max_length=300)
    AQI=models.CharField(max_length=300)
    Prediction=models.CharField(max_length=300)


class air_quality_type_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



