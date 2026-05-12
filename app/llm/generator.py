from google import genai
from app.config import GCP_PROJECT_ID, GCP_LOCATION, LLM_MODEL

class Generator:
    def __init__(self, model_name=LLM_MODEL):
        self.client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT_ID,
            location=GCP_LOCATION
        )
        self.model_name = model_name

    def generate(self, query, contexts):
        context_text = "\n\n".join([c["text"] for c in contexts])

        prompt = f"""
        You are a highly detailed Senior Financial Analyst. 
        Your goal is to answer the user's question with precision using the provided Annual Report context.

        ### Instructions:
        1. **Structured Presentation**: If the data is found in a table, present it as a clean, properly aligned Markdown table.
        2. **Financial Precision**: Ensure all currency signs ($), percentages (%), and parentheticals (representing negative numbers) are preserved exactly.
        3. **Key Highlights**: After presenting a table, provide a bulleted 'Key Takeaways' summary (e.g., 'Net Income increased significantly to $1,485 million in 2024').
        4. **Source Integrity**: If the context contains multiple years (e.g., 2024, 2023), clearly distinguish between them in your answer.
        5. **No Hallucinations**: Use ONLY the provided context. If data is missing, state 'The information is not available in the provided document sections.'

        ### Context:
        {context_text}

        ### Question:
        {query}

        ### Answer:
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "system_instruction": "You are an expert financial assistant focused on accuracy and structured reporting.",
            }
        )
        return response.text

    def generate_stream(self, query, contexts):
        context_text = "\n\n".join([c["text"] for c in contexts])
        prompt = f"""
        You are a highly detailed Senior Financial Analyst. Answer using the context below.
        If tables are present, use Markdown formatting and provide a 'Key Takeaways' summary.

        Context:
        {context_text}

        Question:
        {query}

        Answer:
        """
        
        try:
            # Attempt streaming
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt
            ):
                if chunk and chunk.text:
                    yield chunk.text
        except Exception as e:
            # Fallback to non-streaming if OpenTelemetry/Context issues occur
            print(f"Streaming failed, falling back to non-streaming: {e}")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            yield response.text

if __name__ == "__main__":
    generator = Generator()
    query = "Who is the CEO?"
    contexts = [{"text": "The CEO of IFC is Makhtar Diop."}]
    print(f"Query: {query}")
    print("Response: ", end="")
    for chunk in generator.generate(query, contexts):
        print(chunk, end="", flush=True)
    print()