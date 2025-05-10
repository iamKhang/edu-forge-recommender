from rest_framework import serializers
from .models import Course, User, UserCourseInteraction

class CourseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Course
        fields = '__all__'

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'

class UserCourseInteractionSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserCourseInteraction
        fields = '__all__' 