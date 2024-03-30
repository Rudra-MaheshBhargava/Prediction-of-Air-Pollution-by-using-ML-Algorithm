
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse

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
from Remote_User.models import ClientRegister_Model,air_quality_type,air_quality_type_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')



def Find_Air_Pollution_Predicted_Ratio(request):
    air_quality_type_ratio.objects.all().delete()
    ratio = ""
    kword = 'Poor'
    print(kword)
    obj = air_quality_type.objects.all().filter(Q(Prediction=kword))
    obj1 = air_quality_type.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        air_quality_type_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Very Poor'
    print(kword1)
    obj1 = air_quality_type.objects.all().filter(Q(Prediction=kword1))
    obj11 =air_quality_type.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        air_quality_type_ratio.objects.create(names=kword1, ratio=ratio1)

    ratio12 = ""
    kword12 = 'Severe'
    print(kword12)
    obj12 = air_quality_type.objects.all().filter(Q(Prediction=kword12))
    obj112 = air_quality_type.objects.all()
    count12 = obj12.count();
    count112 = obj112.count();
    ratio12 = (count12 / count112) * 100
    if ratio12 != 0:
        air_quality_type_ratio.objects.create(names=kword12, ratio=ratio12)

    ratio123 = ""
    kword123 = 'Moderate'
    print(kword123)
    obj123 = air_quality_type.objects.all().filter(Q(Prediction=kword123))
    obj1123 = air_quality_type.objects.all()
    count123 = obj123.count();
    count1123 = obj1123.count();
    ratio123 = (count123 / count1123) * 100
    if ratio123 != 0:
        air_quality_type_ratio.objects.create(names=kword123, ratio=ratio123)

    ratio1234 = ""
    kword1234 = 'Satisfactory'
    print(kword1234)
    obj1234 = air_quality_type.objects.all().filter(Q(Prediction=kword1234))
    obj11234 = air_quality_type.objects.all()
    count1234 = obj1234.count();
    count11234 = obj11234.count();
    ratio1234 = (count1234 / count11234) * 100
    if ratio1234 != 0:
        air_quality_type_ratio.objects.create(names=kword1234, ratio=ratio1234)

    obj = air_quality_type_ratio.objects.all()
    return render(request, 'SProvider/Find_Air_Pollution_Predicted_Ratio.html', {'objs': obj})

def View_Air_Pollution_Predicted_Details(request):

    obj = air_quality_type.objects.all().filter()
    return render(request, 'SProvider/View_Air_Pollution_Predicted_Details.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def charts(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def likeschart(request,like_chart):
    charts =air_quality_type_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def Train_Test_Datasets(request):
    detection_accuracy.objects.all().delete()
    df = pd.read_csv('Air_Pollution_Datasets.csv',encoding='latin-1')

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

    labeled = 'labeled_data.csv'
    df.to_csv(labeled, index=False)
    df.to_markdown

    cv = CountVectorizer()

    x = cv.fit_transform(X)

    models = []
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    X_train.shape, X_test.shape, y_train.shape


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
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

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
    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

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
    detection_accuracy.objects.create(names="Decision Tree Classifier", ratio=accuracy_score(y_test, dtcpredict) * 100)


    print("KNeighborsClassifier")
    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier()
    kn.fit(X_train, y_train)
    knpredict = kn.predict(X_test)
    print("ACCURACY")
    print(accuracy_score(y_test, knpredict) * 100)
    print("CLASSIFICATION REPORT")
    print(classification_report(y_test, knpredict))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, knpredict))
    detection_accuracy.objects.create(names="KNeighborsClassifier", ratio=accuracy_score(y_test, knpredict) * 100)

    obj = detection_accuracy.objects.all()
    return render(request, 'SProvider/Train_Test_Datasets.html', {'objs': obj})



def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="PredictedData.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = air_quality_type.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.aid, font_style)
        ws.write(row_num, 1, my_row.City, font_style)
        ws.write(row_num, 2, my_row.Date, font_style)
        ws.write(row_num, 3, my_row.PM2andhalf, font_style)
        ws.write(row_num, 4, my_row.PM10, font_style)
        ws.write(row_num, 5, my_row.NO, font_style)
        ws.write(row_num, 6, my_row.NO2, font_style)
        ws.write(row_num, 7, my_row.Nox, font_style)
        ws.write(row_num, 8, my_row.NH3, font_style)
        ws.write(row_num, 9, my_row.CO, font_style)
        ws.write(row_num, 10, my_row.SO2, font_style)
        ws.write(row_num, 11, my_row.O3, font_style)
        ws.write(row_num, 12, my_row.Benzene, font_style)
        ws.write(row_num, 13, my_row.Toluene, font_style)
        ws.write(row_num, 14, my_row.Xylene, font_style)
        ws.write(row_num, 15, my_row.AQI, font_style)
        ws.write(row_num, 16, my_row.Prediction, font_style)

    wb.save(response)
    return response

















