from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import SkinImageViewSet

router = DefaultRouter()
router.register(r'images', SkinImageViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
