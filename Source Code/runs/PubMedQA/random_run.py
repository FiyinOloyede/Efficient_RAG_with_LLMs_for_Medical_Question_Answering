import warnings, time, re
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from data_preprocessing import load_and_process_pubmedqa
from random_retrieval import random_retriever # random retriever
from models import login_hf, load_model_and_tokenizer
from langchain.prompts import PromptTemplate

warnings.filterwarnings('ignore')

# Login
login_hf("hf_token_here")  # Only if gated 

# Load any of the model in 8-bit ("meta-llama/Meta-Llama-3-8B-Instruct", "deepseek-ai/deepseek-llm-7b-chat",  "mistralai/Mistral-7B-Instruct-v0.3")
model_pipeline = load_model_and_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct", "8bit")

# Load data
df = load_and_process_pubmedqa()

 
# Define Prompt Template with few shot examples
few_shot_examples = [
    {
        "question": "Is galantamine an effective treatment for Alzheimer's disease?",
        "context": "Results from a randomized controlled trial indicated that patients treated with galantamine showed a statistically significant improvement in cognitive scores compared to the placebo group over 24 weeks.",
        "answer": "yes"
    },
    {
        "question": "Does smoking cessation reverse cardiovascular damage completely?",
        "context": "While quitting smoking significantly reduces the risk of cardiovascular events, studies show that some endothelial dysfunction and arterial stiffness may persist for years, suggesting that damage is not fully reversible.",
        "answer": "no"
    },
       {
        "question": "Can vitamin C prevent the common cold?",
        "context": "Several studies have investigated the effect of vitamin C supplementation on the common cold. While some trials suggest it may shorten the duration of colds, the evidence regarding prevention is inconsistent and inconclusive.",
        "answer": "maybe"
    }
]

# Build the few shot prefix string with the new example format.
prefix = ""
for ex in few_shot_examples:
    prefix += f"Question: {ex['question']}\n"
    prefix += f"Background Information: {ex['context']}\n"
    prefix += f"Answer: {ex['answer']}\n\n"

# Build the final prompt with few short prefix
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""
{prefix}
Now this next question:
**Question**: {{question}}

---
**Background Information:**
Based ONLY on the following context, answer the question strictly with a single word: 'yes', 'no', or 'maybe'.

{{context}}

---

**Answer:**
"""
)


# Benchmark for random retriever pipeline
results = []
total = len(df)  # run on all samples

for i, row in df.iterrows():
    if i >= total:
        break

    start = time.time()

    # Retrieve documents
    retrieved_docs = random_retriever(row["question"])
    context = "\n".join([doc.page_content for doc in retrieved_docs])

       # Create the full prompt string
    full_prompt = prompt.format(
        context = context,
        question=row["question"],
    )

    # prefill word count (whitespace-delimited)
    prefill_words = len(full_prompt.strip().split())

    # Call the raw pipeline
    output = model_pipeline(full_prompt, max_new_tokens=10, do_sample=False)[0]["generated_text"]

    # Since the model returns the whole prompt + answer, we isolate the new part
    generated_part = output.replace(full_prompt, "").strip().lower()
 
    latency = time.time() - start

    # Process the generated part to directly extract yes/no/maybe with regex
    found = re.search(r"\b(yes|no|maybe)\b", generated_part)
    predicted = found.group(1) if found else "?"

    # Look at the last line ONLY if the regex failed and the output is not empty
    if predicted == "?":
        lines = output.strip().splitlines()
    # Check if the 'lines' list is not empty before accessing it
        if lines:
            last_line = lines[-1].lower()
            if "yes" in last_line: predicted = "yes"
            elif "no" in last_line: predicted = "no"
            elif "maybe" in last_line: predicted = "maybe"

    results.append({
        "question": row["question"],
        "expected": row["final_decision"],
        "predicted": predicted,
        "is_correct": predicted == row["final_decision"],
        "latency_sec": latency,
        "prefill_words": prefill_words
    })

    if i < 10:
        print(f"\nQuestion: \n{row['question']}")
        print(f"\nRetrieved Context:\n{context}")
        print(f"\nModel Answer: {generated_part}")
        print(f"\nExpected: {row['final_decision']}, Predicted: {predicted}")

# Compile results DataFrame
df_results = pd.DataFrame(results)

# Metrics
accuracy = df_results["is_correct"].mean()
avg_latency = df_results["latency_sec"].mean()
avg_prefill_words = df_results["prefill_words"].mean()

macro_f1 = f1_score(df_results["expected"], df_results["predicted"], average="macro", zero_division=0)
micro_f1 = f1_score(df_results["expected"], df_results["predicted"], average="micro", zero_division=0)
weighted_f1 = f1_score(df_results["expected"], df_results["predicted"], average="weighted", zero_division=0)

# Print summary
print(f"\nMCQ RAG Benchmark completed on {total} samples")
print(f"Accuracy: {accuracy:.2%}")
print(f"Avg latency: {avg_latency:.2f}s")
print(f"Avg prefill words: {avg_prefill_words:.0f}")
print(f"Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
print("\nClassification report:\n", classification_report(df_results["expected"], df_results["predicted"], zero_division=0))

# Save result
df_results.to_csv("Random_llama3_results.csv", index=False)