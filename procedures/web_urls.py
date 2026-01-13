from django.urls import path
from .views import SinistroManualView

urlpatterns = [
    path('manual/', SinistroManualView.as_view(), name='procedures-manual'),
]
