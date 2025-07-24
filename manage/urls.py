# urls.py

from django.urls import path

from manage.user_mgt import views

urlpatterns = [
    path('api/login/', views.login_api, name='api_login'),
    path('api/register/', views.register_api, name='api_register'),
    path('api/upload_app_key/', views.upload_app_key, name='api_upload_app_key'),
    path('api/get_app_key/', views.get_app_key, name='api_get_app_key'),
]