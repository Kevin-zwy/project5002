from django.db import models

# Create your models here.

 #使用Django框架中的模型类来定义一个名为Image的模型
# class Image(models.Model):
#     title = models.CharField(max_length=20)  #字段最长20
#     photo = models.ImageField(upload_to='picture')#保存在项目中的'picture'文件夹下
#    from django.db import models

class Image(models.Model):

    photo = models.ImageField(upload_to='picture')  # 保存在项目中的 'picture' 文件夹下



    def __str__(self):
        return self.photo.name  # 返回图片文件名作为字符串表示
