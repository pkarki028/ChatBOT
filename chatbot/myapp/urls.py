from django.contrib import admin
from django.urls import path,include
from myapp.views import test, index
from django.contrib.auth import views as auth_views
from .import views

from django.contrib.auth.views import LoginView, logout_then_login, LogoutView

app_name="myapp"
urlpatterns = [
    path('bot/', test,name="test"),
    path('',index,name="index"),
    path('login/',auth_views.LoginView.as_view(template_name="accounts/login.html"),name='login'),
    path('logout/',auth_views.LogoutView.as_view(),name='logout'),
    path('signup/',views.SignUp.as_view(),name='signup')
]