import os
from langfuse import Langfuse
from app.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

# Initialize Langfuse client
langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)

def get_langfuse():
    return langfuse

if __name__ == "__main__":
    print(f"Connecting to Langfuse at {LANGFUSE_HOST}...")
    lf = get_langfuse()
    trace = lf.trace(name="test-connection")
    trace.event(name="test-event", input="testing connection")
    print("Test trace sent.")