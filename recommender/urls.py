from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'posts', views.PostViewSet)
router.register(r'users', views.UserViewSet)
router.register(r'interactions', views.UserPostInteractionViewSet)

urlpatterns = [
    path('', include(router.urls)),
] 