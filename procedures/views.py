from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from django.utils import timezone

from .models import ProcedureTemplate, ProcedureInstance, NodeInstance
from .serializers import (
    ProcedureInstanceSerializer,
    ProcedureInstanceDetailSerializer,
    StartProcedureSerializer,
    AnswerNodeSerializer,
)


class StartProcedureView(APIView):
    """
    POST /api/procedures/start/
    Start a new procedure instance from a template.
    
    Request body:
    {
        "template_id": 1
    }
    
    Response:
    {
        "id": 123,
        "template": {...},
        "current_node_id": "1",
        "status": "IN_PROGRESS",
        ...
    }
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        serializer = StartProcedureSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        template_id = serializer.validated_data['template_id']
        template = get_object_or_404(ProcedureTemplate, id=template_id, is_active=True)
        
        # Create new procedure instance
        instance = ProcedureInstance.objects.create(
            template=template,
            started_by=request.user,
            data={}
        )
        
        # Determine first node from template structure
        structure = template.structure
        if isinstance(structure, dict) and 'nodes' in structure and structure['nodes']:
            first_node = structure['nodes'][0]
            instance.current_node_id = first_node.get('id', '1')
            instance.save()
        
        response_serializer = ProcedureInstanceSerializer(instance)
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)


class CurrentNodeView(APIView):
    """
    GET /api/procedures/<instance_id>/current/
    Get the current node information for a procedure instance.
    
    Response:
    {
        "node_id": "1",
        "type": "text",
        "question": "Qual é a placa do veículo?",
        "options": [...],
        "next": "2"
    }
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request, instance_id):
        instance = get_object_or_404(ProcedureInstance, id=instance_id)
        
        # Check if procedure is completed
        if instance.status != 'IN_PROGRESS':
            return Response({
                'status': instance.status,
                'message': 'Procedure is not in progress'
            }, status=status.HTTP_200_OK)
        
        # Get current node from template structure
        current_node_id = instance.current_node_id
        if not current_node_id:
            return Response({
                'error': 'No current node defined'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        structure = instance.template.structure
        if isinstance(structure, dict) and 'nodes' in structure:
            for node in structure['nodes']:
                if node.get('id') == current_node_id:
                    return Response(node, status=status.HTTP_200_OK)
        
        return Response({
            'error': 'Current node not found in template'
        }, status=status.HTTP_404_NOT_FOUND)


class AnswerNodeView(APIView):
    """
    POST /api/procedures/<instance_id>/answer/
    Answer the current node and move to the next one.
    
    Request body:
    {
        "answer": "ABC-1234"  // or any JSON value
    }
    
    Response:
    {
        "success": true,
        "next_node_id": "2",
        "completed": false
    }
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request, instance_id):
        instance = get_object_or_404(ProcedureInstance, id=instance_id)
        
        # Check if procedure is in progress
        if instance.status != 'IN_PROGRESS':
            return Response({
                'error': f'Procedure is {instance.status}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        serializer = AnswerNodeSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        answer_value = serializer.validated_data['answer']
        current_node_id = instance.current_node_id
        
        if not current_node_id:
            return Response({
                'error': 'No current node to answer'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Find current node in template
        structure = instance.template.structure
        current_node = None
        if isinstance(structure, dict) and 'nodes' in structure:
            for node in structure['nodes']:
                if node.get('id') == current_node_id:
                    current_node = node
                    break
        
        if not current_node:
            return Response({
                'error': 'Current node not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # Save the answer as a NodeInstance
        node_instance = NodeInstance.objects.create(
            procedure=instance,
            node_id=current_node_id,
            question=current_node.get('question', ''),
            answer=answer_value,
            answered_by=request.user
        )
        
        # Update procedure data
        data = instance.data or {}
        data[current_node_id] = answer_value
        instance.data = data
        
        # Determine next node
        next_node_id = current_node.get('next')
        
        # Check if there's a next node or if procedure is complete
        if next_node_id:
            instance.current_node_id = next_node_id
            instance.save()
            
            return Response({
                'success': True,
                'next_node_id': next_node_id,
                'completed': False
            }, status=status.HTTP_200_OK)
        else:
            # No next node - procedure is complete
            instance.current_node_id = None
            instance.status = 'COMPLETED'
            instance.completed_at = timezone.now()
            instance.save()
            
            return Response({
                'success': True,
                'next_node_id': None,
                'completed': True
            }, status=status.HTTP_200_OK)


class HistoryView(APIView):
    """
    GET /api/procedures/<instance_id>/history/
    Get the complete history of a procedure instance.
    
    Response:
    {
        "id": 123,
        "template": {...},
        "status": "COMPLETED",
        "nodes": [
            {
                "node_id": "1",
                "question": "...",
                "answer": "...",
                "answered_at": "..."
            },
            ...
        ],
        "data": {...}
    }
    """
    permission_classes = [IsAuthenticated]
    
    def get(self, request, instance_id):
        instance = get_object_or_404(ProcedureInstance, id=instance_id)
        serializer = ProcedureInstanceDetailSerializer(instance)
        return Response(serializer.data, status=status.HTTP_200_OK)
