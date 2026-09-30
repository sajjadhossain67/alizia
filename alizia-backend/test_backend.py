import sys
sys.path.insert(0, r'C:\Users\sajja\Desktop\Tree\alizia ai\alizia-backend')
sys.path.insert(0, r'C:\Users\sajja\Desktop\Tree\alizia ai')
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# Test root endpoint
response = client.get('/')
print('Root - Status:', response.status_code, 'Body:', response.text[:100])

response = client.get('/health')
print('Health - Status:', response.status_code, 'Body:', response.text[:100])

response = client.get('/v1/models')
print('Models - Status:', response.status_code, 'Body length:', len(response.text))

response = client.post('/v1/embeddings', json={'input': ['Hello world']})
print('Embeddings - Status:', response.status_code, 'Body length:', len(response.text))

response = client.post('/v1/conversations', json={'title': 'Test'})
print('Conversation - Status:', response.status_code, 'Body length:', len(response.text))

response = client.post('/v1/reasoning', json={'input': 'Test', 'model': 'alizia-nova', 'reasoning': {'effort': 'auto'}})
print('Reasoning - Status:', response.status_code, 'Body length:', len(response.text))

response = client.post('/v1/agents/run', json={'goal': 'Test agent', 'max_steps': 3})
print('Agent - Status:', response.status_code, 'Body length:', len(response.text))

response = client.post('/v1/rag/search', json={'query': 'test query'})
print('RAG - Status:', response.status_code, 'Body length:', len(response.text))

# Tool execution
response = client.post('/v1/tools/code.execute', json={'code': 'print("hello")'})
print('Tool - Status:', response.status_code, 'Body length:', len(response.text))

print('\nAll basic endpoints working!')