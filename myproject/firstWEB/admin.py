from django.contrib import admin

# Register your models here.
from .models import Image
class imageAdmin(admin.ModelAdmin):  #创建一个名为 imageAdmin 的类，继承自 admin.ModelAdmin。这个类用于定义如何在管理界面中展示和管理 Image 模型
    list_display = ["id","photo"] #个属性指定在列表视图中显示的字段。在这管理界面会显示 photo 字段
    def save_model(self, request, obj, form, change):
        # 在保存模型时自动设置标题为文件名（如果未手动设置）
        if not obj.title and obj.photo:
            obj.title = obj.photo.name  # 使用文件名作为标题
        super().save_model(request, obj, form, change)
admin.site.register(Image, imageAdmin)