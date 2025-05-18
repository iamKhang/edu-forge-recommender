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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the external API URL from environment variable, default to localhost if not set
EXTERNAL_API_URL = os.getenv('EXTERNAL_API_URL', 'http://localhost:8080')
TRAINING_DATA_ENDPOINT = f"{EXTERNAL_API_URL}/api/posts/training-data/all"

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
                logger.info(f"Attempting to connect to external API at: {TRAINING_DATA_ENDPOINT}")
                # Test the API connection first with a timeout of 5 seconds
                response = requests.get(TRAINING_DATA_ENDPOINT, timeout=5)
                response.raise_for_status()  # Raise an exception for bad status codes
                logger.info(f"Successfully connected to external API. Status code: {response.status_code}")
                logger.info(f"Response content: {response.text[:200]}...")  # Log first 200 chars of response
            except requests.exceptions.RequestException as e:
                # Log the specific error
                logger.error(f"Failed to connect to external API: {str(e)}")
                # If API is not available, return simplified response with mock data
                return self._get_mock_recommendations()
            
            # Fetch training data
            logger.info("Fetching training data...")
            training_data = engine.fetch_training_data()
            
            if training_data is None:
                logger.error("No training data returned from fetch_training_data()")
                return Response({
                    'error': 'No training data available',
                    'summary': 'Model training failed due to missing data',
                    'details': {
                        'api_endpoint': TRAINING_DATA_ENDPOINT,
                        'api_response_status': response.status_code if 'response' in locals() else None,
                        'api_response_content': response.text[:200] if 'response' in locals() else None
                    }
                }, status=400)
            
            logger.info(f"Training data fetched successfully. Size: {len(training_data)}")
            
            # Train the model
            logger.info("Starting model training...")
            training_successful = engine.train()
            
            if not training_successful:
                logger.error("Model training failed")
                return Response({
                    'error': 'Model training failed',
                    'summary': 'The model could not be trained with the provided data',
                    'details': {
                        'training_data_size': len(training_data),
                        'possible_reason': 'Insufficient user interactions or invalid data structure'
                    }
                }, status=400)
                
            logger.info("Model training completed successfully")
            
            # Get recommendations for all users
            logger.info("Generating recommendations...")
            all_recommendations = engine.get_all_recommendations()
            if not all_recommendations:
                logger.error("No recommendations generated after training")
                return Response({
                    'error': 'No recommendations available',
                    'summary': 'No recommendations could be generated',
                    'details': {
                        'model_trained': True,
                        'training_data_size': len(training_data)
                    }
                }, status=404)
            
            # Get training statistics
            stats = {
                'num_users': len(engine.user_embeddings) if hasattr(engine, 'user_embeddings') else 0,
                'num_posts': len(engine.post_embeddings) if hasattr(engine, 'post_embeddings') else 0,
                'model_saved': os.path.exists(engine.model_path) if hasattr(engine, 'model_path') else False,
                'embeddings_saved': os.path.exists(engine.embeddings_path) if hasattr(engine, 'embeddings_path') else False
            }
            
            logger.info(f"Process completed successfully. Stats: {stats}")
            return Response({
                'message': 'Model trained and recommendations generated successfully',
                'stats': stats,
                'recommendations': all_recommendations,
                'summary': f"Trained model with {stats['num_users']} users and {stats['num_posts']} posts"
            })
            
        except Exception as e:
            # Get full traceback
            error_details = traceback.format_exc()
            logger.error(f"Error in train_and_recommend: {error_details}")
            
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
