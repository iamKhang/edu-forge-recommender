from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
import tensorflow as tf
import numpy as np
from .models import Post, User, UserPostInteraction
from .serializers import PostSerializer, UserSerializer, UserPostInteractionSerializer
from .recommendation_engine import RecommendationEngine
import os

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

        # Initialize recommendation engine
        engine = RecommendationEngine()
        
        # Try to load existing model
        if not engine.load_model():
            # If model doesn't exist, train it
            engine.train()
        
        # Get user profile and recommendations
        user_profile = engine.get_user_profile(user_id)
        if not user_profile:
            return Response({'error': 'User not found or model not trained'}, status=404)
            
        return Response(user_profile)

    @action(detail=False, methods=['get'])
    def get_all_recommendations(self, request):
        # Initialize recommendation engine
        engine = RecommendationEngine()
        
        # Try to load existing model
        if not engine.load_model():
            # If model doesn't exist, train it
            engine.train()
        
        # Get recommendations for all users
        all_recommendations = engine.get_all_recommendations()
        if not all_recommendations:
            return Response({'error': 'No recommendations available'}, status=404)
            
        return Response(all_recommendations)

    @action(detail=False, methods=['post'])
    def retrain_model(self, request):
        """Retrain the recommendation model"""
        try:
            engine = RecommendationEngine()
            engine.train()
            
            # Get some basic statistics about the model
            stats = {
                'num_users': len(engine.user_embeddings),
                'num_posts': len(engine.post_embeddings),
                'model_saved': os.path.exists(engine.model_path),
                'embeddings_saved': os.path.exists(engine.embeddings_path)
            }
            
            return Response({
                'message': 'Model retrained successfully',
                'stats': stats
            })
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=500)

    @action(detail=False, methods=['post'])
    def train_all_data(self, request):
        """Train the recommendation model on all available data"""
        try:
            engine = RecommendationEngine()
            
            # Fetch all training data
            training_data = engine.fetch_training_data()
            if not training_data:
                return Response({
                    'error': 'No training data available'
                }, status=400)
            
            # Preprocess all data
            post_features, user_interactions = engine.preprocess_data(training_data)
            
            # Train the model
            engine.train()
            
            # Get training statistics
            stats = {
                'num_users': len(engine.user_embeddings),
                'num_posts': len(engine.post_embeddings),
                'num_interactions': sum(len(interactions['views'] | interactions['likes']) 
                                     for interactions in user_interactions.values()),
                'model_saved': os.path.exists(engine.model_path),
                'embeddings_saved': os.path.exists(engine.embeddings_path),
                'training_data_size': len(training_data)
            }
            
            return Response({
                'message': 'Model trained successfully on all data',
                'stats': stats
            })
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=500)
