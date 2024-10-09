
from django.http import HttpResponse

from django.shortcuts import render,redirect
from .models import Image
from .forms import ImageUploadForm
 
# Create your views here.
def index(request):
    data = Image.objects.all()
    context = {
        'data' : data
    }
    return render(request,"display.html", context)
 
def uploadView(request):                                      
    if request.method == 'POST':
        #如果请求方法为 POST，这一行创建一个 `ImageUploadForm` 的实例，通过将 `request.POST` 和 `request.FILES` 传递给它来填充表单数据。
        # `request.POST` 包含用户通过 POST 方法提交的表单数据，而 `request.FILES` 包含用户上传的文件数据。
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid(): #这一行检查表单数据是否有效，即是否通过了表单的验证。
            form.save()#如果表单数据有效，这一行将保存表单数据到数据库中。
            return redirect('index')
    else:
            form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})

