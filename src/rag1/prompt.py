class PromptBuilder:
    """
    Responsible for creating prompts for the LLM.
    """

    def build_prompt(self, query, context):
        """
        Build the final prompt using the user's question
        and retrieved document context.
        """

        prompt = f"""
You are a question-answering assistant for a document-based RAG system.

Answer the question using ONLY the information provided
in the context.

Rules:
1. Do not use outside knowledge.
2. Do not make up information.
3. If the answer is not present in the context, say:
   "I don't have enough information in the provided documents."
4. Give a clear and concise answer.

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

Question:
{query}

Answer:
"""

        return prompt