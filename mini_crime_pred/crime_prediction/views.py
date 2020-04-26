from django.shortcuts import render,redirect
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from  prediction import *
from django.contrib import messages


crime_data=pd.read_csv("C:/Users/Nishant Nimbhorkar/Desktop/MINI PROJECT/dataset_crime_new.csv")
# Create your views here.
def home(request):
    return render(request,'home.html')

def crime_data_analysis(request):
    c1=crime(crime_data)
    c1.preprocessing()
    c1.summary()
    c1.data_day_wise()

    context={'data':c1.data_crime_visuailzation()}
    return render(request,'crime_data_visualization.html',context)



def crime_data_analysis_by_day(request):
    c1=crime(crime_data)
    c1.preprocessing()
    c1.summary()
    context={'data':c1.data_day_wise()}
    return render(request,'data_visualization_day_wise.html',context)


def predecting_crime_type(request):
    source=request.POST.get("source")
    destination=request.POST.get('destination')
    day=request.POST.get('day')
    print("dat-------------")
    if source=='None' or destination=='None' or day=='None':
        messages.error(request, "All  Fields are required for prediction........")
        return redirect("/")
    else:
        try:
            c1=crime(crime_data)
            c1.preprocessing()
            c1.summary()
            c1.data_day_wise()
            c1.data_mapping()
            source_list=c1.getting_api_lattitude_longitude(source)
            destination_list=c1.getting_api_lattitude_longitude(destination)
            print(source)
            print(destination)
            print(day)
            test_data=c1.route_cordinate_api(source_list,destination_list,day)
            context={'data':c1.prediction_modelling(test_data)}
        except:
            messages.error(request, "Something going wrong server problem please try agin by refreshing webpage...")
            return redirect("/")

    return render(request,'visualization_map.html',context)

def bar_crime_data(request):
    data=request.GET.get("data_visual")
    print(data)
    split_data=data.split(",")
    dict={
        'Robbery':int(split_data[0]),
        'Gambling':int(split_data[1]),
        'Accident':int(split_data[2]),
        'Violence':int(split_data[3]),
        'Kidnapping':int(split_data[4]),
        'Murder':int(split_data[5]),
    }
    sorted(dict, key=dict.get, reverse=True)
    context={'data':dict}
    return render(request,'prediction_bar_graph.html',context)





def map(request):
    crime_data=pd.read_csv("C:/Users/Nishant Nimbhorkar/Desktop/MINI PROJECT/dataset_crime_new.csv")

    ################################################### Datset work #############################
    date_data_new=[]
    new_time_stamp=[]
    for i in crime_data['timestamp']:
        if '.' in i.split(" ")[1]:
            # print(i.split(" ")[1])
            new_time=i.split(" ")[1].split('.')[0]+":"+i.split(" ")[1].split('.')[1]
            #print(new_time)
            new_data=i.split(" ")[0]+" "+new_time+":00"
           # print(new_data)
            new_time_stamp.append(new_data)
        else:
            new_time_stamp.append(i+":00")

    crime_data['crime_datetime']=new_time_stamp
    #crime_data['crime_datetime']= crime_data['crime_datetime'].astype('datetime64[ns]')
    day_name= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
    day_of_crime=[]
    for i in crime_data['crime_datetime']:
        date=i.split(" ")[0]
        if '-' in date:
            day = datetime.strptime(date, '%d-%m-%Y').weekday()
            #print(day_name[day])
            day_of_crime.append(day_name[day])
        elif '/' in i:
            day = datetime.strptime(date, '%d/%m/%Y').weekday()
            day_of_crime.append(day_name[day])

    crime_data['crime_day']=day_of_crime



    ################################################# dataset Info  #############################

    print(crime_data.info())
    print(crime_data.head())
    print(crime_data.isnull().sum())
    print(crime_data.describe())

    ############################# Visualization OF Crime rate with respective to days #########################

    Robbery_rate=crime_data[crime_data['Robbery']==1]['crime_day'].value_counts()
    Gambling_rate=crime_data[crime_data['Gambling']==1]['crime_day'].value_counts()
    Accident_rate=crime_data[crime_data['Accident']==1]['crime_day'].value_counts()
    violence_rate=crime_data[crime_data['Violence']==1]['crime_day'].value_counts()
    kidnapping_rate=crime_data[crime_data['Kidnapping']==1]['crime_day'].value_counts()
    murder_rate=crime_data[crime_data['Murder']==1]['crime_day'].value_counts()
    print(Robbery_rate)
    data_frame=pd.DataFrame([Robbery_rate,Gambling_rate,Accident_rate,violence_rate,kidnapping_rate,murder_rate])
    print(data_frame)
    data_frame.index=['Robbery','Gambling','Accident','Violence','Kidnapping','Murder']
    data_frame.plot(kind='bar')


    ################################################# data mapping  #############################
    day_mapping = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6}
    crime_data['crime_day']=crime_data['crime_day'].map(day_mapping)
    crime_data.drop('timestamp', axis=1, inplace=True)


    crime_data["crime_type"]=np.nan





    for i in range(2090):
        if crime_data['Robbery'][i]==1:
           crime_data['crime_type'][i]="Robbery"

        elif crime_data['Gambling'][i]==1:
           crime_data['crime_type'][i]="Gambling"

        elif crime_data['Accident'][i]==1:
           crime_data['crime_type'][i]="Accident"

        elif crime_data['Violence'][i]==1:
           crime_data['crime_type'][i]="Violence"

        elif crime_data['Kidnapping'][i]==1:
           crime_data['crime_type'][i]="Kidnapping"

        elif crime_data['Murder'][i]==1:
           crime_data['crime_type'][i]="Murder"


    Robbery_rate=crime_data[crime_data['crime_type']=='Robbery']['crime_day'].value_counts()
    Gambling_rate=crime_data[crime_data['crime_type']=='Gambling']['crime_day'].value_counts()
    Accident_rate=crime_data[crime_data['crime_type']=='Accident']['crime_day'].value_counts()
    violence_rate=crime_data[crime_data['crime_type']=='Violence']['crime_day'].value_counts()
    kidnapping_rate=crime_data[crime_data['crime_type']=='Kidnapping']['crime_day'].value_counts()
    murder_rate=crime_data[crime_data['crime_type']=='Murder']['crime_day'].value_counts()
    data_frame_vali=pd.DataFrame([Robbery_rate,Gambling_rate,Accident_rate,violence_rate,kidnapping_rate,murder_rate])
    data_frame_vali.index=['Robbery','Gambling','Accident','Violence','Kidnapping','Murder']

    crimetype_mapping = {"Robbery":0,"Gambling":1,"Accident":2,"Violence":3,"Kidnapping":4,"Murder":5}
    crime_data['crime_type']=crime_data['crime_type'].map(crimetype_mapping)



    crime_data.drop('Robbery', axis=1, inplace=True)
    crime_data.drop('Gambling', axis=1, inplace=True)
    crime_data.drop('Accident', axis=1, inplace=True)
    crime_data.drop('Violence', axis=1, inplace=True)
    crime_data.drop('Kidnapping', axis=1, inplace=True)
    crime_data.drop('Murder', axis=1, inplace=True)
    crime_data.drop('crime_datetime', axis=1, inplace=True)


    ######################################### Prediction of Crime Type ########################################

    url = "https://trueway-directions2.p.rapidapi.com/FindDrivingPath"

    querystring = {"origin":"22.718483,75.85474","destination":"22.700081,75.847101"}


    headers = {
        'x-rapidapi-host': "trueway-directions2.p.rapidapi.com",
        'x-rapidapi-key': "8b84f0c448msh19f1df6c542b2fdp14ce81jsnbeb7ca7f849b"
        }
    response = requests.request("GET", url, headers=headers, params=querystring)

    #print(response.text)
    json_data= json.loads(response.text)
    route_data=json_data['route']
    test_data=route_data['geometry']['coordinates']

    for i in test_data:
        i.append(2)

    print(test_data)

    X_train, X_test, y_train, y_test = train_test_split(crime_data, crime_data['crime_type'], test_size=0.2)


    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    clf = KNeighborsClassifier(n_neighbors = 7)
    gnb = GaussianNB()
    rf=RandomForestClassifier(max_depth=2, random_state=0)

    X_train.drop('crime_type',axis=1, inplace=True)
    score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
    score1 = cross_val_score(gnb, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
    score2 = cross_val_score(rf, X_train, y_train, cv=k_fold, n_jobs=1, scoring='accuracy')
    print("KNN-->",round(np.mean(score)*100, 2))
    print("Navie byes-->",round(np.mean(score1)*100, 2))
    print("Random Forest-->",round(np.mean(score2)*100, 2))

    clf.fit(X_train,y_train)
    X_test.drop('crime_type',axis=1, inplace=True)
    tr=[[21.6979,79.8214,1],[22.689,78.45699,1]]
    #prediction=clf.predict(X_test)
    prediction=clf.predict(test_data)
    print(prediction)
    list_of_crime=[]
    list_of_crime1=[]

    crimetype_mapping = {"Robbery":0,"Gambling":1,"Accident":2,"Violence":3,"Kidnapping":4,"Murder":5}

    for i in prediction:
        list_of_crime1.append(i)
        if i==0:
            list_of_crime.append("Robbery")
        elif i==1:
            list_of_crime.append("Gambling")
        elif i==2:
            list_of_crime.append("Accident")
        elif i==3:
            list_of_crime.append("Violence")
        elif i==4:
            list_of_crime.append("Kidnapping")
        elif i==5:
            list_of_crime.append("Murder")
    print(list_of_crime)
    day_number={0:"Sunday",1:"Monday",2:"Tuesday",3:"Wednesday",4:"Thursday",5:"Friday",6:"Saturday"}
    Lattitude=[]
    longitude=[]
    Day=[]

    for i in test_data:
        Lattitude.append(i[0])
        longitude.append(i[1])
        Day.append(day_number[i[2]])


    data_frame_output=pd.DataFrame({'Lattitude':Lattitude,
                                    'longitude':longitude,
                                    'Day':Day,
                                    "Crime_type":list_of_crime,
                                    })
    dict={'Lattitude':Lattitude,
        'longitude':longitude,
        'Day':Day,
        "Crime_type":list_of_crime1,
        }
    context={'data':dict}
    return render(request,'map.html',context)
