import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from datetime import datetime
import os
import json
import logging
import time
import sys
import gc
import psutil
import traceback

# Set up detailed logging
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB

class RecommendationEngine:
    def __init__(self):
        self.model = None
        self.tag_encoder = MultiLabelBinarizer()
        self.content_vectorizer = TfidfVectorizer(max_features=1000)
        self.user_embeddings = {}
        self.post_embeddings = {}
        self.content_embeddings = {}

    # Removed save_model and load_model methods as we don't need to persist the model anymore

    def fetch_training_data(self, max_records=None):
        """Fetch training data from API

        Args:
            max_records: Optional limit on the number of records to process (for testing)
        """
        # Get API URL from environment variable or use default
        external_api_url = os.environ.get('EXTERNAL_API_URL', 'http://localhost:8080')

        # Build the full API endpoint
        api_url = f"{external_api_url}/api/posts/training-data/all"

        logger.info(f"Fetching training data from: {api_url}")
        logger.info(f"Current memory usage: {get_memory_usage():.2f} MB")

        start_time = time.time()

        try:
            # Increase timeout to 30 seconds to handle larger datasets
            logger.info("Sending API request with 30 second timeout")
            response = requests.get(api_url, timeout=30)

            request_time = time.time() - start_time
            logger.info(f"API request completed in {request_time:.2f} seconds with status {response.status_code}")

            if response.status_code == 200:
                # Parse JSON response
                json_start_time = time.time()
                try:
                    data = response.json()
                    json_time = time.time() - json_start_time
                    logger.info(f"JSON parsing completed in {json_time:.2f} seconds")

                    # Log response size
                    response_size_kb = len(response.text) / 1024
                    logger.info(f"Response size: {response_size_kb:.2f} KB")

                    # Check if data is empty or invalid
                    if not data or len(data) == 0:
                        logger.error("Received empty data array from API")
                        return None

                    # Log data details
                    logger.info(f"Successfully fetched {len(data)} training records")

                    # Limit data size if max_records is specified
                    if max_records and len(data) > max_records:
                        logger.info(f"Limiting data to {max_records} records for testing")
                        data = data[:max_records]

                    # Validate data structure
                    valid_records = [post for post in data if 'id' in post and 'userId' in post]
                    invalid_count = len(data) - len(valid_records)

                    if invalid_count > 0:
                        logger.warning(f"Found {invalid_count} records missing required fields (id or userId)")
                        if len(valid_records) == 0:
                            logger.error("No valid records found in the data")
                            return None

                        # Continue with valid records only
                        logger.info(f"Proceeding with {len(valid_records)} valid records")
                        data = valid_records

                    # Log memory usage after data processing
                    logger.info(f"Memory usage after data processing: {get_memory_usage():.2f} MB")

                    return data
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.error(f"Response content (first 500 chars): {response.text[:500]}")
                    return None

            # Handle non-200 responses
            logger.error(f"Failed to fetch training data: HTTP {response.status_code}")
            logger.error(f"Response content (first 500 chars): {response.text[:500] if hasattr(response, 'text') else 'No response text'}")
            return None

        except requests.exceptions.Timeout as e:
            total_time = time.time() - start_time
            logger.error(f"Timeout error fetching training data after {total_time:.2f} seconds: {e}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error fetching training data: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching training data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching training data: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def preprocess_data(self, posts):
        """Preprocess posts data for training"""
        # Extract features
        post_features = []
        user_interactions = {}

        # Prepare content for TF-IDF
        contents = []
        for post in posts:
            contents.append(post.get('content', ''))

        # Fit and transform content
        content_vectors = self.content_vectorizer.fit_transform(contents)

        for i, post in enumerate(posts):
            # Process tags
            tags = post.get('tags', [])

            # Process content
            content = post.get('content', '')
            content_vector = content_vectors[i].toarray()[0]

            # Process user interactions
            views = post.get('views', [])
            likes = post.get('likes', [])

            # Create user interaction matrix
            for view in views:
                user_id = view['userId']
                if user_id not in user_interactions:
                    user_interactions[user_id] = {
                        'views': set(),
                        'likes': set(),
                        'tags': set(),
                        'content_vectors': []
                    }
                user_interactions[user_id]['views'].add(post['id'])
                user_interactions[user_id]['tags'].update(tags)
                user_interactions[user_id]['content_vectors'].append(content_vector)

            for like in likes:
                user_id = like['userId']
                if user_id not in user_interactions:
                    user_interactions[user_id] = {
                        'views': set(),
                        'likes': set(),
                        'tags': set(),
                        'content_vectors': []
                    }
                user_interactions[user_id]['likes'].add(post['id'])
                user_interactions[user_id]['tags'].update(tags)
                user_interactions[user_id]['content_vectors'].append(content_vector)

            # Store post features
            post_features.append({
                'id': post['id'],
                'tags': tags,
                'content': content,
                'content_vector': content_vector
            })

            # Store content embedding
            self.content_embeddings[post['id']] = content_vector

        return post_features, user_interactions

    def build_model(self, num_users, num_posts, embedding_dim=32):
        """Build the recommendation model"""
        # User input
        user_input = tf.keras.layers.Input(shape=(1,), name='user_input')
        user_embedding = tf.keras.layers.Embedding(
            num_users, embedding_dim, name='user_embedding')(user_input)
        user_vec = tf.keras.layers.Flatten(name='user_flatten')(user_embedding)

        # Post input
        post_input = tf.keras.layers.Input(shape=(1,), name='post_input')
        post_embedding = tf.keras.layers.Embedding(
            num_posts, embedding_dim, name='post_embedding')(post_input)
        post_vec = tf.keras.layers.Flatten(name='post_flatten')(post_embedding)

        # Merge layers
        concat = tf.keras.layers.Concatenate()([user_vec, post_vec])
        dense1 = tf.keras.layers.Dense(64, activation='relu')(concat)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)

        # Create model
        model = tf.keras.Model(
            inputs=[user_input, post_input],
            outputs=output
        )

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, max_records=None, epochs=10, batch_size=64):
        """Train the recommendation model

        Args:
            max_records: Optional limit on the number of records to process (for testing)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        total_start_time = time.time()
        logger.info(f"Starting training process. Memory usage: {get_memory_usage():.2f} MB")

        # Fetch and preprocess data
        logger.info("Step 1: Fetching training data...")
        fetch_start_time = time.time()
        posts = self.fetch_training_data(max_records=max_records)
        fetch_time = time.time() - fetch_start_time
        logger.info(f"Data fetching completed in {fetch_time:.2f} seconds")

        # Check if we have valid posts data
        if posts is None or len(posts) == 0:
            logger.error("No valid training data available")
            return False

        # Preprocess data
        logger.info(f"Step 2: Preprocessing {len(posts)} posts...")
        preprocess_start_time = time.time()
        post_features, user_interactions = self.preprocess_data(posts)
        preprocess_time = time.time() - preprocess_start_time
        logger.info(f"Data preprocessing completed in {preprocess_time:.2f} seconds")
        logger.info(f"Memory usage after preprocessing: {get_memory_usage():.2f} MB")

        # Check if we have sufficient interaction data
        if not user_interactions or len(user_interactions) == 0:
            logger.error("No user interactions found in the training data")
            return False

        if not post_features or len(post_features) == 0:
            logger.error("No post features extracted from the training data")
            return False

        # Create user and post mappings
        logger.info("Step 3: Creating user and post mappings...")
        mapping_start_time = time.time()
        user_ids = list(user_interactions.keys())
        post_ids = [post['id'] for post in post_features]

        if len(user_ids) == 0 or len(post_ids) == 0:
            logger.error(f"Insufficient data for training: {len(user_ids)} users, {len(post_ids)} posts")
            return False

        logger.info(f"Training with {len(user_ids)} users and {len(post_ids)} posts")

        user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        post_to_index = {post_id: idx for idx, post_id in enumerate(post_ids)}
        mapping_time = time.time() - mapping_start_time
        logger.info(f"Mapping creation completed in {mapping_time:.2f} seconds")

        # Prepare training data
        logger.info("Step 4: Preparing training data...")
        prep_start_time = time.time()
        X_user = []
        X_post = []
        y = []

        # Track statistics for logging
        users_processed = 0
        users_with_interactions = 0
        total_positive_samples = 0
        total_negative_samples = 0

        for user_id, interactions in user_interactions.items():
            users_processed += 1
            user_idx = user_to_index[user_id]

            # Positive samples (interacted posts)
            interacted_posts = interactions['views'] | interactions['likes']

            if len(interacted_posts) == 0:
                continue  # Skip users with no interactions

            users_with_interactions += 1
            positive_samples = 0

            for post_id in interacted_posts:
                if post_id in post_to_index:  # Make sure the post is in our mapping
                    post_idx = post_to_index[post_id]
                    X_user.append(user_idx)
                    X_post.append(post_idx)
                    y.append(1)
                    positive_samples += 1

            total_positive_samples += positive_samples

            # Negative samples (non-interacted posts)
            non_interacted = set(post_ids) - interacted_posts

            # Skip if there are no non-interacted posts
            if len(non_interacted) == 0:
                continue

            # Choose random non-interacted posts
            try:
                # Limit the number of negative samples to avoid memory issues
                sample_size = min(len(interacted_posts), len(non_interacted), 100)
                non_interacted_sample = np.random.choice(
                    list(non_interacted),
                    size=sample_size,
                    replace=False
                )
            except ValueError as e:
                logger.warning(f"Error sampling non-interacted posts for user {user_id}: {e}")
                continue

            negative_samples = 0
            for post_id in non_interacted_sample:
                post_idx = post_to_index[post_id]
                X_user.append(user_idx)
                X_post.append(post_idx)
                y.append(0)
                negative_samples += 1

            total_negative_samples += negative_samples

            # Log progress for large datasets
            if users_processed % 100 == 0:
                logger.info(f"Processed {users_processed}/{len(user_interactions)} users")
                logger.info(f"Current memory usage: {get_memory_usage():.2f} MB")

        # Check if we have enough training data
        if len(X_user) == 0 or len(X_post) == 0 or len(y) == 0:
            logger.error("No training examples generated")
            return False

        # Log training data statistics
        logger.info(f"Users processed: {users_processed}, Users with interactions: {users_with_interactions}")
        logger.info(f"Positive samples: {total_positive_samples}, Negative samples: {total_negative_samples}")

        # Convert to numpy arrays
        array_start_time = time.time()
        logger.info("Converting to numpy arrays...")
        X_user = np.array(X_user)
        X_post = np.array(X_post)
        y = np.array(y)
        array_time = time.time() - array_start_time
        logger.info(f"Array conversion completed in {array_time:.2f} seconds")

        prep_time = time.time() - prep_start_time
        logger.info(f"Training data preparation completed in {prep_time:.2f} seconds")
        logger.info(f"Created training dataset with {len(X_user)} examples")
        logger.info(f"Memory usage before model building: {get_memory_usage():.2f} MB")

        # Build and train model
        logger.info("Step 5: Building model...")
        build_start_time = time.time()
        self.model = self.build_model(len(user_ids), len(post_ids))
        build_time = time.time() - build_start_time
        logger.info(f"Model building completed in {build_time:.2f} seconds")

        # Run garbage collection before training
        gc.collect()
        logger.info(f"Memory usage after garbage collection: {get_memory_usage():.2f} MB")

        try:
            logger.info(f"Step 6: Training model with {epochs} epochs and batch size {batch_size}...")
            train_start_time = time.time()

            # Use a smaller validation split to reduce memory usage
            history = self.model.fit(
                [X_user, X_post],
                y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=1
            )

            train_time = time.time() - train_start_time
            logger.info(f"Model training completed in {train_time:.2f} seconds")
            logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
            logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

            # Store embeddings for later use
            logger.info("Step 7: Extracting embeddings...")
            embedding_start_time = time.time()

            self.user_embeddings = {
                user_id: self.model.get_layer('user_embedding').get_weights()[0][user_to_index[user_id]]
                for user_id in user_ids
            }

            self.post_embeddings = {
                post_id: self.model.get_layer('post_embedding').get_weights()[0][post_to_index[post_id]]
                for post_id in post_ids
            }

            embedding_time = time.time() - embedding_start_time
            logger.info(f"Embedding extraction completed in {embedding_time:.2f} seconds")

            # No need to save the model anymore
            logger.info("Step 8: Model training completed, no persistence needed")

            total_time = time.time() - total_start_time
            logger.info(f"Total training process completed in {total_time:.2f} seconds")
            logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")

            return True

        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error during model training: {e}")
            logger.error(f"Traceback: {error_details}")
            return False

    def get_content_based_recommendations(self, user_id, num_recommendations=5):
        """Get content-based recommendations for a user"""
        if user_id not in self.user_embeddings:
            return []

        # Get user's content preferences
        user_content_vectors = []
        for post_id in self.post_embeddings.keys():
            if post_id in self.content_embeddings:
                user_content_vectors.append(self.content_embeddings[post_id])

        if not user_content_vectors:
            return []

        # Calculate user's content profile (average of their interacted content)
        user_content_profile = np.mean(user_content_vectors, axis=0)

        # Calculate similarity with all posts
        scores = []
        for post_id, content_vector in self.content_embeddings.items():
            similarity = cosine_similarity(
                user_content_profile.reshape(1, -1),
                content_vector.reshape(1, -1)
            )[0][0]
            scores.append((post_id, float(similarity)))  # Convert to float

        # Sort by similarity and return top recommendations
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:num_recommendations]  # Return list of (post_id, score) tuples

    def get_hybrid_recommendations(self, user_id, num_recommendations=5):
        """Get hybrid recommendations combining collaborative and content-based filtering"""
        if not self.model or user_id not in self.user_embeddings:
            return []

        # Get collaborative filtering recommendations
        collab_recommendations = self.get_recommendations(user_id, num_recommendations)

        # Get content-based recommendations
        content_recommendations = self.get_content_based_recommendations(user_id, num_recommendations)

        # Combine recommendations with weights
        combined_scores = {}

        # Weight for collaborative filtering
        collab_weight = 0.6
        # Weight for content-based filtering
        content_weight = 0.4

        # Add collaborative filtering scores
        for post_id, score in collab_recommendations:
            combined_scores[post_id] = score * collab_weight

        # Add content-based filtering scores
        for post_id, score in content_recommendations:
            if post_id in combined_scores:
                combined_scores[post_id] += score * content_weight
            else:
                combined_scores[post_id] = score * content_weight

        # Sort by combined score
        sorted_recommendations = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_recommendations[:num_recommendations]  # Return list of (post_id, score) tuples

    def get_recommendations(self, user_id, num_recommendations=5):
        """Get post recommendations for a user using collaborative filtering"""
        if not self.model or user_id not in self.user_embeddings:
            return []

        # Get user embedding
        user_embedding = self.user_embeddings[user_id]

        # Calculate similarity scores with all posts
        scores = []
        for post_id, post_embedding in self.post_embeddings.items():
            similarity = np.dot(user_embedding, post_embedding) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(post_embedding)
            )
            scores.append((post_id, float(similarity)))  # Convert to float

        # Sort by similarity and return top recommendations
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:num_recommendations]  # Return list of (post_id, score) tuples

    def get_user_profile(self, user_id):
        """Get user profile with their interests and behavior patterns"""
        if not self.model or user_id not in self.user_embeddings:
            return None

        # Get user embedding
        user_embedding = self.user_embeddings[user_id]

        # Calculate similarity with all other users
        user_similarities = []
        for other_user_id, other_embedding in self.user_embeddings.items():
            if other_user_id != user_id:
                similarity = np.dot(user_embedding, other_embedding) / (
                    np.linalg.norm(user_embedding) * np.linalg.norm(other_embedding)
                )
                user_similarities.append((other_user_id, float(similarity)))  # Convert to float

        # Sort by similarity
        user_similarities.sort(key=lambda x: x[1], reverse=True)

        # Convert numpy arrays to lists and ensure all values are JSON serializable
        return {
            'user_id': user_id,
            'embedding': user_embedding.tolist(),  # Convert numpy array to list
            'similar_users': [
                {'user_id': uid, 'similarity': sim}  # sim is already float
                for uid, sim in user_similarities[:5]
            ],
            'collaborative_recommendations': [
                {'post_id': pid, 'score': float(score)}
                for pid, score in self.get_recommendations(user_id)
            ],
            'content_based_recommendations': [
                {'post_id': pid, 'score': float(score)}
                for pid, score in self.get_content_based_recommendations(user_id)
            ],
            'hybrid_recommendations': [
                {'post_id': pid, 'score': float(score)}
                for pid, score in self.get_hybrid_recommendations(user_id)
            ]
        }

    def get_all_recommendations(self):
        """Get recommendations for all users"""
        if not self.model:
            return []

        all_recommendations = []
        for user_id in self.user_embeddings.keys():
            user_profile = self.get_user_profile(user_id)
            if user_profile:
                all_recommendations.append(user_profile)

        return all_recommendations