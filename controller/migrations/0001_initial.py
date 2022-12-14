# Generated by Django 4.1.2 on 2022-11-02 08:19

import controller.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ProcessLog",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("user", models.CharField(max_length=50)),
                ("img_path", models.CharField(max_length=500)),
                ("finished", models.CharField(max_length=500)),
                ("finished_time", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="Requested",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("user", models.CharField(max_length=50)),
                (
                    "image",
                    models.ImageField(
                        blank=True,
                        null=True,
                        upload_to=controller.models.upload_to,
                        verbose_name="image",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Responsed",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("user", models.CharField(max_length=50)),
                ("contractId", models.CharField(max_length=50)),
                ("imagePath", models.CharField(max_length=500)),
                ("result", models.CharField(max_length=2000)),
            ],
        ),
    ]
