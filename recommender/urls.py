from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'posts', views.PostViewSet, basename='post')
router.register(r'users', views.UserViewSet, basename='user')
router.register(r'interactions', views.UserPostInteractionViewSet, basename='interaction')

urlpatterns = [
    path('', include(router.urls)),
]