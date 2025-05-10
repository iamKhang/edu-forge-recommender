from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
import tensorflow as tf
import numpy as np
from .models import Post, User, UserPostInteraction
from .serializers import PostSerializer, UserSerializer, UserPostInteractionSerializer
from .recommendation_engine import RecommendationEngine
import requests
import os
import traceback
import json

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

    @action(detail=False, methods=['get', 'post'])
    def train_and_recommend(self, request):
        """Train model and get recommendations for all users"""
        try:
            # Initialize recommendation engine
            engine = RecommendationEngine()
            
            # Check if the external API is available
            try:
                # Test the API connection first with a timeout of 2 seconds
                response = requests.get('http://localhost:8080/api/posts/training-data/all', timeout=2)
                if response.status_code != 200:
                    # If API is not available, return simplified response with mock data
                    return self._get_mock_recommendations()
            except requests.exceptions.RequestException as e:
                # If API is not available, return simplified response with mock data
                return self._get_mock_recommendations()
            
            # Fetch training data
            training_data = engine.fetch_training_data()
            if not training_data:
                return Response({
                    'error': 'No training data available',
                    'summary': 'Model training failed due to missing data'
                }, status=400)
            
            # Train the model
            engine.train()
            
            # Get recommendations for all users
            all_recommendations = engine.get_all_recommendations()
            if not all_recommendations:
                return Response({
                    'error': 'No recommendations available',
                    'summary': 'No recommendations could be generated'
                }, status=404)
            
            # Get training statistics
            stats = {
                'num_users': len(engine.user_embeddings),
                'num_posts': len(engine.post_embeddings),
                'model_saved': os.path.exists(engine.model_path),
                'embeddings_saved': os.path.exists(engine.embeddings_path)
            }
            
            return Response({
                'message': 'Model trained and recommendations generated successfully',
                'stats': stats,
                'recommendations': all_recommendations,
                'summary': f"Trained model with {stats['num_users']} users and {stats['num_posts']} posts"
            })
            
        except Exception as e:
            # Get full traceback
            error_details = traceback.format_exc()
            print(f"Error in train_and_recommend: {error_details}")
            
            return Response({
                'error': str(e),
                'details': error_details,
                'summary': 'An error occurred during model training and recommendation generation'
            }, status=500)
            
    def _get_mock_recommendations(self):
        """Return mock recommendations for testing when API is unavailable"""
        mock_data = {
            'message': 'MOCK DATA: External API not available, using mock recommendations',
            'stats': {
                'num_users': 3,
                'num_posts': 5,
                'model_saved': False,
                'embeddings_saved': False
            },
            'summary': 'Using mock data because external API is unavailable',
            'recommendations': [
                {
                    'user_id': 'user1',
                    'embedding': [0.1, 0.2, 0.3],
                    'similar_users': [
                        {'user_id': 'user2', 'similarity': 0.85}
                    ],
                    'collaborative_recommendations': [
                        {'post_id': 'post1', 'score': 0.92},
                        {'post_id': 'post3', 'score': 0.75}
                    ],
                    'content_based_recommendations': [
                        {'post_id': 'post2', 'score': 0.88},
                        {'post_id': 'post5', 'score': 0.65}
                    ],
                    'hybrid_recommendations': [
                        {'post_id': 'post1', 'score': 0.90},
                        {'post_id': 'post2', 'score': 0.85}
                    ]
                },
                {
                    'user_id': 'user2',
                    'embedding': [0.2, 0.3, 0.4],
                    'similar_users': [
                        {'user_id': 'user1', 'similarity': 0.85}
                    ],
                    'collaborative_recommendations': [
                        {'post_id': 'post2', 'score': 0.90},
                        {'post_id': 'post4', 'score': 0.72}
                    ],
                    'content_based_recommendations': [
                        {'post_id': 'post3', 'score': 0.82},
                        {'post_id': 'post5', 'score': 0.68}
                    ],
                    'hybrid_recommendations': [
                        {'post_id': 'post2', 'score': 0.88},
                        {'post_id': 'post3', 'score': 0.82}
                    ]
                },
                {
                    'user_id': 'user3',
                    'embedding': [0.3, 0.4, 0.5],
                    'similar_users': [
                        {'user_id': 'user2', 'similarity': 0.78}
                    ],
                    'collaborative_recommendations': [
                        {'post_id': 'post3', 'score': 0.88},
                        {'post_id': 'post1', 'score': 0.70}
                    ],
                    'content_based_recommendations': [
                        {'post_id': 'post4', 'score': 0.85},
                        {'post_id': 'post2', 'score': 0.62}
                    ],
                    'hybrid_recommendations': [
                        {'post_id': 'post3', 'score': 0.86},
                        {'post_id': 'post4', 'score': 0.80}
                    ]
                }
            ]
        }
        
        return Response(mock_data)
