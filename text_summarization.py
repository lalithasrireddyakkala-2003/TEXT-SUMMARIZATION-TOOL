from transformers import pipeline

def summarize_text(text):
    # Load pre-trained summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Summarize
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Example lengthy article
input_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned 
with the interactions between computers and human language, in particular how to program computers to process and analyze 
large amounts of natural language data. The goal is a computer capable of 'understanding' the contents of documents, 
including the contextual nuances of the language within them. The technology can then accurately extract information 
and insights contained in the documents as well as categorize and organize the documents themselves.
"""

# Display results
print("\n📝 Original Text:\n")
print(input_text)

print("\n🔍 Summary:\n")
print(summarize_text(input_text))
