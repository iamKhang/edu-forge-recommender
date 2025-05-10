from rest_framework import serializers
from .models import Post, User, UserPostInteraction

class PostSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = '__all__'

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'

class UserPostInteractionSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserPostInteraction
        fields = '__all__' 