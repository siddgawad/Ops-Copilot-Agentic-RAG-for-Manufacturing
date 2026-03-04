import math 

corpus = [
    "the cnc machien spindle is broken",
    "replace the spindle bearing on the cnc",
    "the safety door on the machine is open"
]

target_term = "spindle"

#TODO 1: Calculate TF for target_term in document 1

# TODO 1: Calculate TF (Term Frequency) for Document 1
doc1 = corpus[0]
doc1_words = doc1.split(" ")

term_count = doc1_words.count(target_term)  # How many times does 'spindle' appear?
total_words = len(doc1_words)               # How many words total in this doc?

tf = term_count / total_words
print(f"TF for Doc 1: {term_count} / {total_words} = {tf}")

# TODO 2: Calculate IDF (Inverse Document Frequency) across ALL documents
total_docs = len(corpus)

# Count how many documents contain the word 'spindle'
docs_with_term = 0
for doc in corpus:
    if target_term in doc.split(" "):
        docs_with_term += 1

# The formula: natural log of (Total Docs / Docs containing term)
idf = math.log(total_docs / docs_with_term)
print(f"IDF across corpus: ln({total_docs} / {docs_with_term}) = {idf:.4f}")

# TODO 3: Calculate the final TF-IDF Score
tf_idf = tf * idf
print(f"Final TF-IDF Score for '{target_term}' in Doc 1: {tf_idf:.4f}")