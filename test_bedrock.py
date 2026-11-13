import boto3
from botocore.exceptions import ClientError

MODEL_ID = "global.anthropic.claude-sonnet-4-6"

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

try:
    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": "Reply with exactly: Bedrock connection successful"}
                ]
            }
        ],
        inferenceConfig={
            "maxTokens": 100,
            "temperature": 0
        }
    )

    print(response["output"]["message"]["content"][0]["text"])

except ClientError as error:
    print("Bedrock access failed:")
    print(error.response["Error"]["Code"])
    print(error.response["Error"]["Message"])