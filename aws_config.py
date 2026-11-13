import os
from dotenv import load_dotenv

load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE") or None

# Amazon Bedrock - Claude Sonnet 5.5
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "global.anthropic.claude-sonnet-4-6"
)

# Database query result limit
MAX_RESULT_ROWS = max(
    1,
    min(int(os.getenv("MAX_RESULT_ROWS", "1000")), 1000)
)