class PromptBuilder:
    """
    Responsible for creating prompts for the LLM.
    """

    def build_prompt(self, query, context):
        """
        Combine the user's question and retrieved context
        into the final prompt.
        """

        prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context,
say:

"I don't have enough information in the provided documents."

Do not make up information.

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

Question:
{query}

Answer:
"""

        return prompt