from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
import tensorflow as tf
import numpy as np
from .models import Post, User, UserPostInteraction
from .serializers import PostSerializer, UserSerializer, UserPostInteractionSerializer
from .recommendation_engine import RecommendationEngine, get_memory_usage
import requests
import os
import traceback
import json
import logging
import time
import socket

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Use the production API URL for training data
TRAINING_DATA_ENDPOINT = "http://eduforge.io.vn:8081/api/v1/posts/training-data/all"

# Configuration constants
MAX_RECORDS_FOR_TESTING = 1000  # Limit data size for testing
TRAINING_TIMEOUT = 600  # 10 minutes timeout for training
API_REQUEST_TIMEOUT = 30  # 30 seconds timeout for API requests

# Create your views here.

class PostViewSet(viewsets.ViewSet):
    def list(self, request):
        return Response({"message": "Posts API is not available in memory-only mode"})

class UserViewSet(viewsets.ViewSet):
    def list(self, request):
        return Response({"message": "Users API is not available in memory-only mode"})

class UserPostInteractionViewSet(viewsets.ViewSet):
    def list(self, request):
        return Response({"message": "Interactions API is not available in memory-only mode"})

    @action(detail=False, methods=['get', 'post'])
    def train_and_recommend(self, request):
        """Train model and get recommendations for all users"""
        start_time = time.time()
        logger.info(f"Starting train_and_recommend endpoint. Memory usage: {get_memory_usage():.2f} MB")

        # Get parameters from request
        test_mode = request.query_params.get('test_mode', 'false').lower() == 'true'
        max_records = int(request.query_params.get('max_records', MAX_RECORDS_FOR_TESTING if test_mode else 0))
        epochs = int(request.query_params.get('epochs', 5 if test_mode else 10))
        batch_size = int(request.query_params.get('batch_size', 32 if test_mode else 64))

        # Log request parameters
        logger.info(f"Request parameters: test_mode={test_mode}, max_records={max_records}, epochs={epochs}, batch_size={batch_size}")

        # Log system information
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        logger.info(f"System info: Hostname={hostname}, IP={ip_address}")

        try:
            # Initialize recommendation engine
            logger.info("Initializing recommendation engine...")
            engine = RecommendationEngine()

            # Check if the external API is available
            try:
                logger.info(f"Attempting to connect to external API at: {TRAINING_DATA_ENDPOINT}")
                # Test the API connection first with a timeout
                response = requests.get(TRAINING_DATA_ENDPOINT, timeout=API_REQUEST_TIMEOUT)
                response.raise_for_status()  # Raise an exception for bad status codes
                logger.info(f"Successfully connected to external API. Status code: {response.status_code}")
                logger.info(f"Response content size: {len(response.text)} bytes")
                logger.info(f"Response content preview: {response.text[:200]}...")  # Log first 200 chars of response
            except requests.exceptions.Timeout as e:
                logger.error(f"Timeout connecting to external API after {API_REQUEST_TIMEOUT} seconds: {str(e)}")
                return Response({
                    'error': f'API timeout after {API_REQUEST_TIMEOUT} seconds',
                    'summary': 'The external API took too long to respond',
                    'details': {
                        'api_endpoint': TRAINING_DATA_ENDPOINT,
                        'timeout': API_REQUEST_TIMEOUT,
                        'error_type': 'Timeout',
                        'error_message': str(e)
                    }
                }, status=504)  # Gateway Timeout
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error to external API: {str(e)}")
                return Response({
                    'error': 'Connection error to external API',
                    'summary': 'Could not establish connection to the external API',
                    'details': {
                        'api_endpoint': TRAINING_DATA_ENDPOINT,
                        'error_type': 'ConnectionError',
                        'error_message': str(e),
                        'possible_causes': [
                            'Network connectivity issues',
                            'External service is down',
                            'Incorrect API URL configuration'
                        ]
                    }
                }, status=503)  # Service Unavailable
            except requests.exceptions.RequestException as e:
                # Log the specific error
                logger.error(f"Failed to connect to external API: {str(e)}")
                # If API is not available, return simplified response with mock data
                logger.info("Falling back to mock recommendations")
                return self._get_mock_recommendations()

            # Fetch training data
            logger.info("Fetching training data...")
            training_data = engine.fetch_training_data(max_records=max_records)

            if training_data is None:
                logger.error("No training data returned from fetch_training_data()")
                return Response({
                    'error': 'No training data available',
                    'summary': 'Model training failed due to missing data',
                    'details': {
                        'api_endpoint': TRAINING_DATA_ENDPOINT,
                        'api_response_status': response.status_code if 'response' in locals() else None,
                        'api_response_size': len(response.text) if 'response' in locals() and hasattr(response, 'text') else None,
                        'memory_usage': f"{get_memory_usage():.2f} MB"
                    }
                }, status=400)

            logger.info(f"Training data fetched successfully. Size: {len(training_data)} records")

            # Train the model with parameters
            logger.info(f"Starting model training with epochs={epochs}, batch_size={batch_size}...")
            training_start_time = time.time()
            training_successful = engine.train(max_records=max_records, epochs=epochs, batch_size=batch_size)
            training_time = time.time() - training_start_time
            logger.info(f"Model training completed in {training_time:.2f} seconds. Success: {training_successful}")

            if not training_successful:
                logger.error("Model training failed")
                return Response({
                    'error': 'Model training failed',
                    'summary': 'The model could not be trained with the provided data',
                    'details': {
                        'training_data_size': len(training_data),
                        'training_time': f"{training_time:.2f} seconds",
                        'memory_usage': f"{get_memory_usage():.2f} MB",
                        'possible_reason': 'Insufficient user interactions or invalid data structure'
                    }
                }, status=400)

            logger.info("Model training completed successfully")

            # Get recommendations for all users
            logger.info("Generating recommendations...")
            recommendation_start_time = time.time()
            all_recommendations = engine.get_all_recommendations()
            recommendation_time = time.time() - recommendation_start_time
            logger.info(f"Recommendation generation completed in {recommendation_time:.2f} seconds")

            if not all_recommendations:
                logger.error("No recommendations generated after training")
                return Response({
                    'error': 'No recommendations available',
                    'summary': 'No recommendations could be generated',
                    'details': {
                        'model_trained': True,
                        'training_data_size': len(training_data),
                        'training_time': f"{training_time:.2f} seconds",
                        'memory_usage': f"{get_memory_usage():.2f} MB"
                    }
                }, status=404)

            # Get training statistics
            stats = {
                'num_users': len(engine.user_embeddings) if hasattr(engine, 'user_embeddings') else 0,
                'num_posts': len(engine.post_embeddings) if hasattr(engine, 'post_embeddings') else 0,
                'model_saved': False,  # We don't save models anymore
                'embeddings_saved': False,  # We don't save embeddings anymore
                'training_time': f"{training_time:.2f} seconds",
                'recommendation_time': f"{recommendation_time:.2f} seconds",
                'memory_usage': f"{get_memory_usage():.2f} MB",
                'test_mode': test_mode,
                'epochs': epochs,
                'batch_size': batch_size
            }

            total_time = time.time() - start_time
            logger.info(f"Process completed successfully in {total_time:.2f} seconds. Stats: {stats}")

            return Response({
                'message': 'Model trained and recommendations generated successfully',
                'stats': stats,
                'recommendations': all_recommendations,
                'summary': f"Trained model with {stats['num_users']} users and {stats['num_posts']} posts in {total_time:.2f} seconds"
            })

        except Exception as e:
            # Get full traceback
            error_details = traceback.format_exc()
            logger.error(f"Error in train_and_recommend: {error_details}")

            total_time = time.time() - start_time
            memory_usage = get_memory_usage()

            return Response({
                'error': str(e),
                'details': error_details,
                'summary': 'An error occurred during model training and recommendation generation',
                'diagnostics': {
                    'total_time': f"{total_time:.2f} seconds",
                    'memory_usage': f"{memory_usage:.2f} MB",
                    'hostname': hostname,
                    'ip_address': ip_address
                }
            }, status=500)

    @action(detail=False, methods=['get'])
    def test_connection(self, request):
        """Test the connection to the external API without running the full training process"""
        start_time = time.time()
        logger.info(f"Starting test_connection endpoint. Memory usage: {get_memory_usage():.2f} MB")

        # Log system information
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        logger.info(f"System info: Hostname={hostname}, IP={ip_address}")

        # Get timeout parameter from request
        timeout = int(request.query_params.get('timeout', API_REQUEST_TIMEOUT))

        try:
            logger.info(f"Attempting to connect to external API at: {TRAINING_DATA_ENDPOINT}")
            logger.info(f"Using timeout of {timeout} seconds")

            # Test the API connection
            response = requests.get(TRAINING_DATA_ENDPOINT, timeout=timeout)

            # Calculate response time
            response_time = time.time() - start_time

            # Get response details
            status_code = response.status_code
            response_size = len(response.text)
            content_type = response.headers.get('Content-Type', 'unknown')

            logger.info(f"API connection successful. Status: {status_code}, Time: {response_time:.2f}s, Size: {response_size} bytes")

            # Try to parse as JSON if content type is application/json
            data_sample = None
            record_count = None

            if 'application/json' in content_type:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        record_count = len(data)
                        data_sample = data[:2] if record_count > 0 else []
                    else:
                        data_sample = data
                except json.JSONDecodeError:
                    logger.warning("Response content is not valid JSON despite Content-Type")

            return Response({
                'status': 'success',
                'message': f'Successfully connected to external API in {response_time:.2f} seconds',
                'details': {
                    'api_endpoint': TRAINING_DATA_ENDPOINT,
                    'status_code': status_code,
                    'response_time': f"{response_time:.2f} seconds",
                    'response_size': f"{response_size} bytes",
                    'content_type': content_type,
                    'record_count': record_count,
                    'data_sample': data_sample,
                    'memory_usage': f"{get_memory_usage():.2f} MB",
                    'hostname': hostname,
                    'ip_address': ip_address
                }
            })

        except requests.exceptions.Timeout as e:
            response_time = time.time() - start_time
            logger.error(f"Timeout connecting to external API after {response_time:.2f} seconds: {str(e)}")

            return Response({
                'status': 'error',
                'error': f'API timeout after {response_time:.2f} seconds',
                'message': 'The external API took too long to respond',
                'details': {
                    'api_endpoint': TRAINING_DATA_ENDPOINT,
                    'timeout': timeout,
                    'response_time': f"{response_time:.2f} seconds",
                    'error_type': 'Timeout',
                    'error_message': str(e),
                    'memory_usage': f"{get_memory_usage():.2f} MB",
                    'hostname': hostname,
                    'ip_address': ip_address
                }
            }, status=504)  # Gateway Timeout

        except requests.exceptions.ConnectionError as e:
            response_time = time.time() - start_time
            logger.error(f"Connection error to external API: {str(e)}")

            return Response({
                'status': 'error',
                'error': 'Connection error to external API',
                'message': 'Could not establish connection to the external API',
                'details': {
                    'api_endpoint': TRAINING_DATA_ENDPOINT,
                    'response_time': f"{response_time:.2f} seconds",
                    'error_type': 'ConnectionError',
                    'error_message': str(e),
                    'memory_usage': f"{get_memory_usage():.2f} MB",
                    'hostname': hostname,
                    'ip_address': ip_address,
                    'possible_causes': [
                        'Network connectivity issues',
                        'External service is down',
                        'Incorrect API URL configuration'
                    ]
                }
            }, status=503)  # Service Unavailable

        except Exception as e:
            response_time = time.time() - start_time
            error_details = traceback.format_exc()
            logger.error(f"Error testing API connection: {error_details}")

            return Response({
                'status': 'error',
                'error': str(e),
                'message': 'An error occurred while testing the API connection',
                'details': {
                    'api_endpoint': TRAINING_DATA_ENDPOINT,
                    'response_time': f"{response_time:.2f} seconds",
                    'error_details': error_details,
                    'memory_usage': f"{get_memory_usage():.2f} MB",
                    'hostname': hostname,
                    'ip_address': ip_address
                }
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
