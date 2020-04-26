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


class crime():
    def __init__(self,crime_data):
        self.crime_data=crime_data

################################################### Datset work #############################
    def preprocessing(self):
        date_data_new=[]
        new_time_stamp=[]
        for i in self.crime_data['timestamp']:
            if '.' in i.split(" ")[1]:
                # print(i.split(" ")[1])
                new_time=i.split(" ")[1].split('.')[0]+":"+i.split(" ")[1].split('.')[1]
                #print(new_time)
                new_data=i.split(" ")[0]+" "+new_time+":00"
               # print(new_data)
                new_time_stamp.append(new_data)
            else:
                new_time_stamp.append(i+":00")

        self.crime_data['crime_datetime']=new_time_stamp
        #crime_data['crime_datetime']= crime_data['crime_datetime'].astype('datetime64[ns]')
        day_name= ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
        day_of_crime=[]
        for i in self.crime_data['crime_datetime']:
            date=i.split(" ")[0]
            if '-' in date:
                day = datetime.strptime(date, '%d-%m-%Y').weekday()
                #print(day_name[day])
                day_of_crime.append(day_name[day])
            elif '/' in i:
                day = datetime.strptime(date, '%d/%m/%Y').weekday()
                day_of_crime.append(day_name[day])

        self.crime_data['crime_day']=day_of_crime


################################################# dataset Info  #############################
    def summary(self):
        print(self.crime_data.info())
        print(self.crime_data.head())
        print(self.crime_data.isnull().sum())
        print(self.crime_data.describe())

    def data_day_wise(self):
        print("dataploting------------------")
        Robbery_rate=self.crime_data[self.crime_data['Robbery']==1]['crime_day'].value_counts()
        Gambling_rate=self.crime_data[self.crime_data['Gambling']==1]['crime_day'].value_counts()
        Accident_rate=self.crime_data[self.crime_data['Accident']==1]['crime_day'].value_counts()
        violence_rate=self.crime_data[self.crime_data['Violence']==1]['crime_day'].value_counts()
        kidnapping_rate=self.crime_data[self.crime_data['Kidnapping']==1]['crime_day'].value_counts()
        murder_rate=self.crime_data[self.crime_data['Murder']==1]['crime_day'].value_counts()
        data_frame=pd.DataFrame([Robbery_rate,Gambling_rate,Accident_rate,violence_rate,kidnapping_rate,murder_rate])
        data_frame.index=['Robbery','Gambling','Accident','Violence','Kidnapping','Murder']
        data_frame.plot(kind='bar')
        data_frame["Wednesday"].fillna(0, inplace = True)
        data_frame["Thursday"].fillna(0, inplace = True)
        data_frame["Monday"].fillna(0, inplace = True)
        data_frame["Saturday"].fillna(0, inplace = True)
        data_frame["Sunday"].fillna(0, inplace = True)
        data_frame["Tuesday"].fillna(0, inplace = True)
        print(data_frame)

        dict={'Wednesday':{'Robbery':data_frame['Wednesday'][0],
                            'Gambling':data_frame['Wednesday'][1],
                            'Accident':data_frame['Wednesday'][2],
                            'Violence':data_frame['Wednesday'][3],
                            'Kidnapping':data_frame['Wednesday'][4],
                            'Murder':data_frame['Wednesday'][5],
                           },
            'Thursday':{'Robbery':data_frame['Thursday'][0],
                                'Gambling':data_frame['Thursday'][1],
                                'Accident':data_frame['Thursday'][2],
                                'Violence':data_frame['Thursday'][3],
                                'Kidnapping':data_frame['Thursday'][4],
                                'Murder':data_frame['Thursday'][5],
                               },
            'Monday':{'Robbery':data_frame['Monday'][0],
                                'Gambling':data_frame['Monday'][1],
                                'Accident':data_frame['Monday'][2],
                                'Violence':data_frame['Monday'][3],
                                'Kidnapping':data_frame['Monday'][4],
                                'Murder':data_frame['Monday'][5],
                               },

            'Saturday':{'Robbery':data_frame['Saturday'][0],
                                'Gambling':data_frame['Saturday'][1],
                                'Accident':data_frame['Saturday'][2],
                                'Violence':data_frame['Saturday'][3],
                                'Kidnapping':data_frame['Saturday'][4],
                                'Murder':data_frame['Saturday'][5],
                               },
            'Sunday':{'Robbery':data_frame['Sunday'][0],
                                'Gambling':data_frame['Sunday'][1],
                                'Accident':data_frame['Sunday'][2],
                                'Violence':data_frame['Sunday'][3],
                                'Kidnapping':data_frame['Sunday'][4],
                                'Murder':data_frame['Sunday'][5],
                               },
            'Tuesday':{'Robbery':data_frame['Tuesday'][0],
                                'Gambling':data_frame['Tuesday'][1],
                                'Accident':data_frame['Tuesday'][2],
                                'Violence':data_frame['Tuesday'][3],
                                'Kidnapping':data_frame['Tuesday'][4],
                                'Murder':data_frame['Tuesday'][5],
                               },
            }
        return dict

    def data_crime_visuailzation(self):
        Robbery_rate=(self.crime_data['Robbery']==1).sum()
        Gambling_rate=(self.crime_data['Gambling']==1).sum()
        Accident_rate=(self.crime_data['Accident']==1).sum()
        violence_rate=(self.crime_data['Violence']==1).sum()
        kidnapping_rate=(self.crime_data['Kidnapping']==1).sum()
        murder_rate=(self.crime_data['Murder']==1).sum()
        dict={
            'Robbery':Robbery_rate,
            'Gambling':Gambling_rate,
            'Accident':Accident_rate,
            'Violence':violence_rate,
            'Kidnapping':kidnapping_rate,
            'Murder':murder_rate,
        }
        return dict

    def data_mapping(self):
        day_mapping = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6}
        self.crime_data['crime_day']=self.crime_data['crime_day'].map(day_mapping)
        self.crime_data.drop('timestamp', axis=1, inplace=True)


        self.crime_data["crime_type"]=np.nan

        for i in range(2090):
            if self.crime_data['Robbery'][i]==1:
               self.crime_data['crime_type'][i]="Robbery"

            elif self.crime_data['Gambling'][i]==1:
               self.crime_data['crime_type'][i]="Gambling"

            elif self.crime_data['Accident'][i]==1:
               self.crime_data['crime_type'][i]="Accident"

            elif self.crime_data['Violence'][i]==1:
               self.crime_data['crime_type'][i]="Violence"

            elif self.crime_data['Kidnapping'][i]==1:
               self.crime_data['crime_type'][i]="Kidnapping"

            elif self.crime_data['Murder'][i]==1:
               self.crime_data['crime_type'][i]="Murder"


        Robbery_rate=self.crime_data[self.crime_data['crime_type']=='Robbery']['crime_day'].value_counts()
        Gambling_rate=self.crime_data[self.crime_data['crime_type']=='Gambling']['crime_day'].value_counts()
        Accident_rate=self.crime_data[self.crime_data['crime_type']=='Accident']['crime_day'].value_counts()
        violence_rate=self.crime_data[self.crime_data['crime_type']=='Violence']['crime_day'].value_counts()
        kidnapping_rate=self.crime_data[self.crime_data['crime_type']=='Kidnapping']['crime_day'].value_counts()
        murder_rate=self.crime_data[self.crime_data['crime_type']=='Murder']['crime_day'].value_counts()
        data_frame_vali=pd.DataFrame([Robbery_rate,Gambling_rate,Accident_rate,violence_rate,kidnapping_rate,murder_rate])
        data_frame_vali.index=['Robbery','Gambling','Accident','Violence','Kidnapping','Murder']

        crimetype_mapping = {"Robbery":0,"Gambling":1,"Accident":2,"Violence":3,"Kidnapping":4,"Murder":5}
        self.crime_data['crime_type']=self.crime_data['crime_type'].map(crimetype_mapping)



        self.crime_data.drop('Robbery', axis=1, inplace=True)
        self.crime_data.drop('Gambling', axis=1, inplace=True)
        self.crime_data.drop('Accident', axis=1, inplace=True)
        self.crime_data.drop('Violence', axis=1, inplace=True)
        self.crime_data.drop('Kidnapping', axis=1, inplace=True)
        self.crime_data.drop('Murder', axis=1, inplace=True)
        self.crime_data.drop('crime_datetime', axis=1, inplace=True)


    def getting_api_lattitude_longitude(self,place_name):
        url = "https://trueway-places.p.rapidapi.com/FindPlaceByText"

        querystring = {"language":"en","text":place_name+",indore,"+"india"}

        headers = {
            'x-rapidapi-host': "trueway-places.p.rapidapi.com",
            'x-rapidapi-key': "8b84f0c448msh19f1df6c542b2fdp14ce81jsnbeb7ca7f849b"
            }

        response = requests.request("GET", url, headers=headers, params=querystring)

        json_data= json.loads(response.text)
        print(json_data)
        route_data=json_data['results']
        list_lat_lng=[]
        for i in route_data:
            list_lat_lng.append(i['location']['lat'])
            list_lat_lng.append(i['location']['lng'])

        print(list_lat_lng)
        return list_lat_lng

    def route_cordinate_api(self,source,destination,day):
        url = "https://trueway-directions2.p.rapidapi.com/FindDrivingPath"

        querystring = {"origin":''+str(source[0])+','+str(source[1])
                        ,"destination":''+str(destination[0])+','+str(destination[1])
                        }


        headers = {
            'x-rapidapi-host': "trueway-directions2.p.rapidapi.com",
            'x-rapidapi-key': "8b84f0c448msh19f1df6c542b2fdp14ce81jsnbeb7ca7f849b"
            }
        response = requests.request("GET", url, headers=headers, params=querystring)

        #print(response.text)
        json_data= json.loads(response.text)
        route_data=json_data['route']
        test_data=route_data['geometry']['coordinates']
        day_option_dict = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6}

        for i in test_data:
            i.append(day_option_dict[day])

        print(test_data)
        return test_data



    def prediction_modelling(self,test_data):

        X_train, X_test, y_train, y_test = train_test_split(self.crime_data, self.crime_data['crime_type'], test_size=0.2)


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
        return dict
'''
crime_data=pd.read_csv("C:/Users/Nishant Nimbhorkar/Desktop/MINI PROJECT/dataset_crime_new.csv")
c1=crime(crime_data)
c1.preprocessing()
c1.summary()
c1.data_day_wise()
c1.data_mapping()
c1.prediction_modelling()'''
