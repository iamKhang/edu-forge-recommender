from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'courses', views.CourseViewSet)
router.register(r'users', views.UserViewSet)
router.register(r'interactions', views.UserCourseInteractionViewSet)

urlpatterns = [
    path('', include(router.urls)),
] 