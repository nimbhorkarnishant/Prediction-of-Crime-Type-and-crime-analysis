from django.urls import path
from .import views


urlpatterns = [
     path('',views.home,name='home'),
     path('crime_data_analysis/',views.crime_data_analysis,name='crime_data_analysis'),
     path('crime_data_analysis_by_day/',views.crime_data_analysis_by_day,name='crime_data_analysis_by_day'),
     path('predecting_crime_type/',views.predecting_crime_type,name='prediction'),
     path('bar_crime_data/',views.bar_crime_data,name='bar_crime_data'),
     path('map/',views.map,name='map')

 ]
