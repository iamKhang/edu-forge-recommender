from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
import tensorflow as tf
import numpy as np
from .models import Course, User, UserCourseInteraction
from .serializers import CourseSerializer, UserSerializer, UserCourseInteractionSerializer

# Create your views here.

class CourseViewSet(viewsets.ModelViewSet):
    queryset = Course.objects.all()
    serializer_class = CourseSerializer

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

class UserCourseInteractionViewSet(viewsets.ModelViewSet):
    queryset = UserCourseInteraction.objects.all()
    serializer_class = UserCourseInteractionSerializer

    @action(detail=False, methods=['get'])
    def get_recommendations(self, request):
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({'error': 'user_id is required'}, status=400)

        # TODO: Implement recommendation logic using TensorFlow
        # This is a placeholder for the actual recommendation system
        recommended_courses = Course.objects.all()[:5]
        serializer = CourseSerializer(recommended_courses, many=True)
        return Response(serializer.data)
