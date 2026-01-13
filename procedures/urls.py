from django.urls import path
from .views import StartProcedureView, CurrentNodeView, AnswerNodeView, HistoryView, SinistroManualView

urlpatterns = [
    path('manual/', SinistroManualView.as_view(), name='procedures-manual'),
    path('start/', StartProcedureView.as_view(), name='procedure_start'),
    path('<int:instance_id>/current/', CurrentNodeView.as_view(), name='procedure_current'),
    path('<int:instance_id>/answer/', AnswerNodeView.as_view(), name='procedure_answer'),
    path('<int:instance_id>/history/', HistoryView.as_view(), name='procedure_history'),
]
