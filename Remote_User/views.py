from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl

from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,air_quality_type,air_quality_type_ratio

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city, address=address, gender=gender)
        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_AirPollution(request):
    se=''
    if request.method == "POST":
        kword = request.POST.get('keyword')
        if request.method == "POST":

            aid = request.POST.get('aid')
            City= request.POST.get('City')
            Date= request.POST.get('Date')
            PM2andhalf= request.POST.get('PM2andhalf')
            PM10= request.POST.get('PM10')
            NO= request.POST.get('NO')
            NO2= request.POST.get('NO2')
            Nox= request.POST.get('NOX')
            NH3= request.POST.get('NH3')
            CO= request.POST.get('CO')
            SO2= request.POST.get('SO2')
            O3= request.POST.get('O3')
            Benzene= request.POST.get('Benzene')
            Toluene= request.POST.get('Toluene')
            Xylene= request.POST.get('Xylene')
            AQI= request.POST.get('AQI')

        df = pd.read_csv('Air_Pollution_Datasets.csv')

        def apply_results(results):
            if (results == 'Poor'):
                return 0
            elif (results == 'Very Poor'):
                return 1
            elif (results == 'Severe'):
                return 2
            elif (results == 'Moderate'):
                return 3
            elif (results == 'Satisfactory'):
                return 4
            elif (results == 'Good'):
                return 5

        df['results'] = df['AQI_Bucket'].apply(apply_results)

        X = df['MID']
        y = df['results']

        cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))

        x = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB

        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print("ACCURACY")
        print(naivebayes)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_nb))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print("ACCURACY")
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('SVM', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('LogisticRegression', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))
        models.append(('DecisionTreeClassifier', dtc))


        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        aid = [aid]
        vector1 = cv.transform(aid).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = str(pred.replace("]", ""))

        prediction = int(pred1)

        if (prediction == 0):
            val = 'Poor'

        elif (prediction == 1):
            val = 'Very Poor'

        elif (prediction == 2):
            val = 'Severe'

        elif (prediction == 3):
            val= 'Moderate'

        elif (prediction == 4):
            val= 'Satisfactory'

        elif (prediction == 5):
            val= 'Good'


        print(prediction)
        print(val)

        air_quality_type.objects.create(aid=aid,
        City=City,
        Date=Date,
        PM2andhalf=PM2andhalf,
        PM10=PM10,
        NO=NO,
        NO2=NO2,
        Nox=Nox,
        NH3=NH3,
        CO=CO,
        SO2=SO2,
        O3=O3,
        Benzene=Benzene,
        Toluene=Toluene,
        Xylene=Xylene,
        AQI=AQI,
        Prediction=val)

        return render(request, 'RUser/Predict_AirPollution.html',{'objs': val})
    return render(request, 'RUser/Predict_AirPollution.html')





