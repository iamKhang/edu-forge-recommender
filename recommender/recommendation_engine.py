import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from datetime import datetime
import os
import pickle

class RecommendationEngine:
    def __init__(self):
        self.model = None
        self.tag_encoder = MultiLabelBinarizer()
        self.content_vectorizer = TfidfVectorizer(max_features=1000)
        self.user_embeddings = {}
        self.post_embeddings = {}
        self.model_path = 'recommender/model/recommendation_model'
        self.embeddings_path = 'recommender/model/embeddings.pkl'
        
        # Create model directory if it doesn't exist
        os.makedirs('recommender/model', exist_ok=True)
        
    def save_model(self):
        """Save the trained model and embeddings"""
        if self.model:
            self.model.save(self.model_path)
            embeddings = {
                'user_embeddings': self.user_embeddings,
                'post_embeddings': self.post_embeddings
            }
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
                
    def load_model(self):
        """Load the trained model and embeddings"""
        if os.path.exists(self.model_path) and os.path.exists(self.embeddings_path):
            self.model = tf.keras.models.load_model(self.model_path)
            with open(self.embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
                self.user_embeddings = embeddings['user_embeddings']
                self.post_embeddings = embeddings['post_embeddings']
            return True
        return False

    def fetch_training_data(self):
        """Fetch training data from API"""
        response = requests.get('http://localhost:8080/api/posts/training-data/all')
        if response.status_code == 200:
            return response.json()
        return []

    def preprocess_data(self, posts):
        """Preprocess posts data for training"""
        # Extract features
        post_features = []
        user_interactions = {}
        
        for post in posts:
            # Process tags
            tags = post.get('tags', [])
            
            # Process content
            content = post.get('content', '')
            
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
                        'tags': set()
                    }
                user_interactions[user_id]['views'].add(post['id'])
                user_interactions[user_id]['tags'].update(tags)
            
            for like in likes:
                user_id = like['userId']
                if user_id not in user_interactions:
                    user_interactions[user_id] = {
                        'views': set(),
                        'likes': set(),
                        'tags': set()
                    }
                user_interactions[user_id]['likes'].add(post['id'])
                user_interactions[user_id]['tags'].update(tags)
            
            # Store post features
            post_features.append({
                'id': post['id'],
                'tags': tags,
                'content': content
            })
        
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

    def train(self):
        """Train the recommendation model"""
        # Fetch and preprocess data
        posts = self.fetch_training_data()
        post_features, user_interactions = self.preprocess_data(posts)
        
        # Create user and post mappings
        user_ids = list(user_interactions.keys())
        post_ids = [post['id'] for post in post_features]
        
        user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        post_to_index = {post_id: idx for idx, post_id in enumerate(post_ids)}
        
        # Prepare training data
        X_user = []
        X_post = []
        y = []
        
        for user_id, interactions in user_interactions.items():
            user_idx = user_to_index[user_id]
            
            # Positive samples (interacted posts)
            for post_id in interactions['views'] | interactions['likes']:
                post_idx = post_to_index[post_id]
                X_user.append(user_idx)
                X_post.append(post_idx)
                y.append(1)
            
            # Negative samples (non-interacted posts)
            non_interacted = set(post_ids) - (interactions['views'] | interactions['likes'])
            # Chọn ngẫu nhiên các bài viết chưa tương tác
            non_interacted = np.random.choice(
                list(non_interacted),
                size=min(len(interactions['views'] | interactions['likes']), len(non_interacted)),
                replace=False
            )
            for post_id in non_interacted:
                post_idx = post_to_index[post_id]
                X_user.append(user_idx)
                X_post.append(post_idx)
                y.append(0)
        
        # Convert to numpy arrays
        X_user = np.array(X_user)
        X_post = np.array(X_post)
        y = np.array(y)
        
        # Build and train model
        self.model = self.build_model(len(user_ids), len(post_ids))
        
        self.model.fit(
            [X_user, X_post],
            y,
            epochs=10,
            batch_size=64,
            validation_split=0.2
        )
        
        # Store embeddings for later use
        self.user_embeddings = {
            user_id: self.model.get_layer('user_embedding').get_weights()[0][user_to_index[user_id]]
            for user_id in user_ids
        }
        
        self.post_embeddings = {
            post_id: self.model.get_layer('post_embedding').get_weights()[0][post_to_index[post_id]]
            for post_id in post_ids
        }
        
        # Save the trained model and embeddings
        self.save_model()

    def get_recommendations(self, user_id, num_recommendations=5):
        """Get post recommendations for a user"""
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
            scores.append((post_id, similarity))
        
        # Sort by similarity and return top recommendations
        scores.sort(key=lambda x: x[1], reverse=True)
        return [post_id for post_id, _ in scores[:num_recommendations]]

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
                user_similarities.append((other_user_id, similarity))
        
        # Sort by similarity
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'user_id': user_id,
            'embedding': user_embedding.tolist(),
            'similar_users': [
                {'user_id': uid, 'similarity': float(sim)}
                for uid, sim in user_similarities[:5]
            ],
            'recommended_posts': self.get_recommendations(user_id)
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