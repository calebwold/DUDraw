import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def handler(event, context):
    """Netlify serverless function for status check"""
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, OPTIONS'
    }
    
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    api_key_configured = bool(os.environ.get("OPENAI_API_KEY"))
    
    return {
        'statusCode': 200,
        'headers': headers,
        'body': json.dumps({
            "status": "ready" if api_key_configured else "not_configured",
            "api_key_configured": api_key_configured,
            "data_source": "du_draw_functions_data.py"
        })
    }

