from rest_framework import serializers
from .models import SkinImage

class SkinImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = SkinImage
        fields = ['image', 'uploaded_at']
