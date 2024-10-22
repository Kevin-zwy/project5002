from torchvision import transforms
from PIL import Image
from rest_framework.response import Response
from rest_framework import status, viewsets
from .models import SkinImage
from .serializers import SkinImageSerializer
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models

import sys
import os
path = 'D:\\project\\5002\\myproject\\api'
sys.path.append(os.path.abspath(path))
from .implementation_final import predict_image_voting,CustomDenseNet121,CustomMobileNetV2,CustomResNet50


                


class_info = {
    'Acne and Rosacea Photos': {
        'explanation': 'Acne and rosacea are common skin conditions characterized by redness and pustules on the face, chest, and back.',
        'suggestion': 'It is advisable to use over-the-counter acne treatments and keep the skin clean.',
    },
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions': {
        'explanation': 'Actinic keratosis and other lesions may be precursors to malignant conditions, often found on skin exposed to the sun for prolonged periods.',
        'suggestion': 'It is recommended to consult a dermatologist for timely examination and treatment.',
    },
    'Atopic Dermatitis Photos': {
        'explanation': 'Atopic dermatitis is a common allergic skin condition, especially in infants, characterized by dry and itchy skin.',
        'suggestion': 'Keep the skin moisturized, avoid irritants, and use gentle skincare products.',
    },
    'Bullous Disease Photos': {
        'explanation': 'Bullous diseases are a class of skin conditions that cause blisters on the skin, often accompanied by itching and pain.',
        'suggestion': 'It is advised to avoid scratching, keep the blistered area clean, and consult a doctor.',
    },
    'Cellulitis Impetigo and other Bacterial Infections': {
        'explanation': 'Cellulitis and impetigo are bacterial infections characterized by redness, pain, and fever.',
        'suggestion': 'Prompt medical attention is required, and antibiotics may be necessary.',
    },
    'Eczema Photos': {
        'explanation': 'Eczema is a chronic inflammatory skin condition characterized by redness, itching, and dryness.',
        'suggestion': 'Keep the skin moisturized and avoid irritants and allergens.',
    },
    'Exanthems and Drug Eruptions': {
        'explanation': 'Drug eruptions are skin reactions caused by drug allergies, typically presenting as rashes.',
        'suggestion': 'Stop taking the suspected medication immediately and consult a doctor.',
    },
    'Hair Loss Photos Alopecia and other Hair Diseases': {
        'explanation': 'Hair loss can be caused by various factors, including genetics, stress, and nutritional deficiencies.',
        'suggestion': 'It is recommended to consult a specialist for relevant examination and treatment.',
    },
    'Herpes HPV and other STDs Photos': {
        'explanation': 'Genital herpes and human papillomavirus (HPV) are common sexually transmitted diseases that may present as sores or lesions in the genital area.',
        'suggestion': 'Seek medical attention promptly for testing and consultation.',
    },
    'Light Diseases and Disorders of Pigmentation': {
        'explanation': 'Pigmentation disorders may lead to uneven skin coloration, often due to UV exposure and genetic factors.',
        'suggestion': 'Use sunscreen to protect the skin and consult a dermatologist if necessary.',
    },
    'Lupus and other Connective Tissue diseases': {
        'explanation': 'Lupus is an autoimmune disease that may affect the skin and other organs, presenting with rashes and joint pain.',
        'suggestion': 'Regular check-ups are recommended, following the doctor\'s treatment plan.',
    },
    'Melanoma Skin Cancer Nevi and Moles': {
        'explanation': 'Melanoma is a malignant skin cancer typically presenting as changing moles or new pigment lesions.',
        'suggestion': 'It is advisable to seek medical attention promptly for skin examinations.',
    },
    'Nail Fungus and other Nail Disease': {
        'explanation': 'Nail fungus infections can cause discoloration, thickening, and brittleness of the nails.',
        'suggestion': 'Keep the nails dry and clean, using antifungal medications for treatment.',
    },
    'Poison Ivy Photos and other Contact Dermatitis': {
        'explanation': 'Contact dermatitis is caused by skin contact with irritants or allergens, usually presenting as redness and itching.',
        'suggestion': 'Avoid known allergens and use antihistamines to relieve symptoms.',
    },
    'Psoriasis pictures Lichen Planus and related diseases': {
        'explanation': 'Psoriasis is a chronic skin condition characterized by red patches and silvery scales.',
        'suggestion': 'It is advisable to use medications to control the condition and consult a specialist.',
    },
    'Scabies Lyme Disease and other Infestations and Bites': {
        'explanation': 'Scabies and Lyme disease are caused by parasites or bacterial infections, presenting with itching and rashes.',
        'suggestion': 'It is recommended to seek medical attention promptly for appropriate treatment.',
    },
    'Seborrheic Keratoses and other Benign Tumors': {
        'explanation': 'Seborrheic keratosis is a benign skin tumor that is usually harmless but may affect appearance.',
        'suggestion': 'Regular skin checks are recommended, and removal may be considered if necessary.',
    },
    'Systemic Disease': {
        'explanation': 'Systemic diseases may affect the whole body and present with various skin symptoms.',
        'suggestion': 'Comprehensive examination is recommended to determine potential causes.',
    },
    'Tinea Ringworm Candidiasis and other Fungal Infections': {
        'explanation': 'Fungal infections such as ringworm and candidiasis can cause redness and itching of the skin.',
        'suggestion': 'Use antifungal medications for treatment and keep the skin dry.',
    },
    'Urticaria Hives': {
        'explanation': 'Urticaria is caused by allergic reactions and presents as red, raised welts on the skin.',
        'suggestion': 'Avoid known allergens and consider antihistamines for relief.',
    },
    'Vascular Tumors': {
        'explanation': 'Vascular tumors are benign tumors that typically appear as red or purple growths on the skin.',
        'suggestion': 'Consult a doctor if necessary to determine if treatment is required.',
    },
    'Vasculitis Photos': {
        'explanation': 'Vasculitis is an inflammatory condition affecting blood vessels, potentially causing skin spots or ulcers.',
        'suggestion': 'Seek medical attention promptly for detailed examination and treatment.',
    },
    'Warts Molluscum and other Viral Infections': {
        'explanation': 'Warts and molluscum contagiosum are benign skin conditions caused by viruses, commonly found on the skin and mucous membranes.',
        'suggestion': 'Treatment is usually not necessary, but consult a doctor if needed.',
    },
}

class SkinImageViewSet(viewsets.ModelViewSet):
    queryset = SkinImage.objects.all()
    serializer_class = SkinImageSerializer

    def create(self, request, *args, **kwargs):
        # ���л��ϴ���ͼƬ
        serializer = self.get_serializer(data=request.data)
        # return Response({'result': "HELLO"}, status=status.HTTP_201_CREATED) 
        if serializer.is_valid():
            serializer.save()
            
            # ��ȡ�ϴ���ͼƬ
            image = serializer.instance.image
            img_path = image.path
            img = Image.open(img_path)                      
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            class_names = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Bullous Disease Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions', 'Hair Loss Photos Alopecia and other Hair Diseases', 'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 'Nail Fungus and other Nail Disease', 'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', 'Scabies Lyme Disease and other Infestations and Bites', 'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']
            num_classes = 23
            model_resnet = CustomResNet50(num_classes=num_classes).to(device)
            model_densenet = CustomDenseNet121(num_classes=num_classes).to(device)
            model_mobilenetv2 = CustomMobileNetV2(num_classes=num_classes).to(device)

            # model_resnet = CustomResNet50(num_classes=num_classes)


            loaded_model_resnet = torch.load(r'D:\project\5002\myproject\detection\resnet50_skin_disease_model.pth', weights_only=False)
            loaded_model_densenet = torch.load(r'D:\project\5002\myproject\detection\DenseNet121_skin_disease_model_29.pth', weights_only=False)
            loaded_model_mobilenetv2 = torch.load(r'D:\project\5002\myproject\detection\mobilenet__skin_disease_model.pth', weights_only=False)
            # Check if the loaded file is a dictionary (state_dict) or the model itself
            if isinstance(loaded_model_resnet, dict):
                model_resnet.load_state_dict(loaded_model_resnet)
                model_densenet.load_state_dict(loaded_model_densenet)
                model_mobilenetv2.load_state_dict(loaded_model_mobilenetv2)
            else:
                # If it's the model, assign it directly
                model_resnet = loaded_model_resnet
                model_densenet = loaded_model_densenet
                loaded_model_mobilenetv2 = loaded_model_mobilenetv2

            model_resnet.to(device)
            model_densenet.to(device)
            loaded_model_mobilenetv2.to(device)

            models=[model_resnet,model_densenet,model_mobilenetv2]
                # Upload image or specify image path
            image_path = r'C:\Users\123\Desktop\project5001\微信图片_20241016162227.png' #path
            
            prediction = predict_image_voting(img_path, models, device, class_names)
            # 获取解释和建议
            explanation = class_info[prediction]['explanation']
            suggestion = class_info[prediction]['suggestion']
            return Response({
                'result': "YES", 
                'prediction': prediction, 
                'explanation': explanation, 
    '           suggestion': suggestion
            }, status=status.HTTP_201_CREATED)
        else:
            print(serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            # �����������
        #     return Response({'result': "YES"}, status=status.HTTP_201_CREATED)
        # else:
        #     print(serializer.errors)
        #     return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
