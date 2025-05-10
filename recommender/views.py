from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
import tensorflow as tf
import numpy as np
from .models import Post, User, UserPostInteraction
from .serializers import PostSerializer, UserSerializer, UserPostInteractionSerializer
from .recommendation_engine import RecommendationEngine

# Create your views here.

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

class UserPostInteractionViewSet(viewsets.ModelViewSet):
    queryset = UserPostInteraction.objects.all()
    serializer_class = UserPostInteractionSerializer

    @action(detail=False, methods=['get'])
    def get_recommendations(self, request):
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({'error': 'user_id is required'}, status=400)

        # Initialize and train recommendation engine
        engine = RecommendationEngine()
        engine.train()
        
        # Get user profile and recommendations
        user_profile = engine.get_user_profile(user_id)
        if not user_profile:
            return Response({'error': 'User not found or model not trained'}, status=404)
            
        return Response(user_profile)

    @action(detail=False, methods=['get'])
    def get_all_recommendations(self, request):
        # Initialize and train recommendation engine
        engine = RecommendationEngine()
        engine.train()
        
        # Get recommendations for all users
        all_recommendations = engine.get_all_recommendations()
        if not all_recommendations:
            return Response({'error': 'No recommendations available'}, status=404)
            
        return Response(all_recommendations)
