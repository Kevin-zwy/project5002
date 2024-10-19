import torch
from torchvision import transforms
from PIL import Image
from rest_framework.response import Response
from rest_framework import status, viewsets
from .models import SkinImage
from .serializers import SkinImageSerializer


class SkinImageViewSet(viewsets.ModelViewSet):
    queryset = SkinImage.objects.all()
    serializer_class = SkinImageSerializer

    def create(self, request, *args, **kwargs):
        # 序列化上传的图片
        serializer = self.get_serializer(data=request.data)
        # return Response({'result': "HELLO"}, status=status.HTTP_201_CREATED) 
        if serializer.is_valid():
            serializer.save()
            
            # 获取上传的图片
            image = serializer.instance.image
            img_path = image.path
            img = Image.open(img_path)

            # 返回推理结果
            return Response({'result': "YES"}, status=status.HTTP_201_CREATED)
        else:
            print(serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
