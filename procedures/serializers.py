from rest_framework import serializers
from .models import ProcedureTemplate, ProcedureInstance, NodeInstance


class ProcedureTemplateSerializer(serializers.ModelSerializer):
    """Serializer for ProcedureTemplate model"""
    
    class Meta:
        model = ProcedureTemplate
        fields = ['id', 'name', 'description', 'version', 'structure', 'is_active', 
                  'created_at', 'updated_at', 'created_by']
        read_only_fields = ['id', 'created_at', 'updated_at', 'created_by']


class NodeInstanceSerializer(serializers.ModelSerializer):
    """Serializer for NodeInstance model"""
    
    class Meta:
        model = NodeInstance
        fields = ['id', 'node_id', 'question', 'answer', 'answered_at', 'answered_by']
        read_only_fields = ['id', 'answered_at', 'answered_by']


class ProcedureInstanceSerializer(serializers.ModelSerializer):
    """Serializer for ProcedureInstance model"""
    template_name = serializers.CharField(source='template.name', read_only=True)
    nodes = NodeInstanceSerializer(many=True, read_only=True)
    
    class Meta:
        model = ProcedureInstance
        fields = ['id', 'template', 'template_name', 'current_node_id', 'status', 
                  'started_at', 'completed_at', 'started_by', 'data', 'nodes']
        read_only_fields = ['id', 'started_at', 'completed_at', 'started_by']


class ProcedureInstanceDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer with full node history"""
    template = ProcedureTemplateSerializer(read_only=True)
    nodes = NodeInstanceSerializer(many=True, read_only=True)
    
    class Meta:
        model = ProcedureInstance
        fields = ['id', 'template', 'current_node_id', 'status', 
                  'started_at', 'completed_at', 'started_by', 'data', 'nodes']
        read_only_fields = ['id', 'started_at', 'completed_at', 'started_by']


class StartProcedureSerializer(serializers.Serializer):
    """Serializer for starting a new procedure"""
    template_id = serializers.IntegerField(required=True)


class AnswerNodeSerializer(serializers.Serializer):
    """Serializer for answering a node"""
    answer = serializers.JSONField(required=True)
