# Generated by Django 4.2 on 2024-10-08 15:17

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("firstWEB", "0003_image_title"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="image",
            name="title",
        ),
    ]