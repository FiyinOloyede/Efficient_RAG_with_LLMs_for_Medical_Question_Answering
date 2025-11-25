import warnings, time, re
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from data_preprocessing import load_and_process_medqa_us
from bm25_retrieval import bm25_retriever # bm25 retriever
from dense_retrieval import dense_retriever_gtr, dense_retriever_sbb # dense retrievers
from dense_retrieval import hybrid_with_rerank_gtr, hybrid_with_rerank_sbb # hybrid retrievers
from models import login_hf, load_model_and_tokenizer
from langchain.prompts import PromptTemplate

warnings.filterwarnings('ignore')

# Login
login_hf("hf_token_here")  # Only if gated 

# Load any of the model in 8-bit ("meta-llama/Meta-Llama-3-8B-Instruct", "deepseek-ai/deepseek-llm-7b-chat",  "mistralai/Mistral-7B-Instruct-v0.3")
model_pipeline = load_model_and_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct", "8bit")

# Load data
df = load_and_process_medqa_us()

 
# Define Prompt Template with few shot examples
few_shot_examples = [
    {
      "question": "What is the typical cause of iron‑deficiency anemia?",
      "opa": "Autoimmune destruction of red cells",
      "opb": "Vitamin B12 deficiency",
      "opc": "Blood loss due to ulcers",
      "opd": "Genetic hemoglobin disorder",
      "ope": "Aortic Dissection",
      "answer": "C",
      "context": "Iron‑deficiency anemia typically occurs due to chronic blood loss—for example, from gastrointestinal ulcers or heavy menstruation."
    },
    {
      "question": "A 55-year-old male presents with sudden onset of severe, tearing chest pain radiating to his back. His blood pressure is 200/120 mmHg in the right arm and 160/90 mmHg in the left arm. What is the most likely diagnosis?",
      "opa": "Myocardial Infarction",
      "opb": "Pulmonary Embolism",
      "opc": "Aortic Dissection",
      "opd": "Pericarditis",
      "ope": "Blood loss due to ulcers",
      "answer": "C",
      "context": "Aortic dissection is a medical emergency that involves a tear in the inner layer of the aorta. It classically presents with sudden, severe chest or back pain, often described as 'tearing' in nature. A significant blood pressure differential between the arms is a key clinical finding."
    },
    {
      "question": "Which medication is first‑line for type 2 diabetes?",
      "opa": "Metformin",
      "opb": "Insulin",
      "opc": "Sulfonylurea",
      "opd": "GLP‑1 agonist",
      "ope": "Myocardial Infarction",
      "answer": "A",
      "context": "Clinical guidelines recommend metformin as first‑line treatment for type 2 diabetes unless contraindicated."
    }
]

# Build the few‑shot prefix
prefix = ""
for ex in few_shot_examples:
    prefix += f"Question: {ex['question']}\nA. {ex['opa']}\nB. {ex['opb']}\nC. {ex['opc']}\nD. {ex['opd']}\nE. {ex['ope']}\n"
    prefix += f"Background Information: {ex['context']}\n"
    prefix += f"Answer: {ex['answer']}\n\n"

# Build the final prompt with few short prefix
prompt = PromptTemplate(
    input_variables=["context", "question", "opa", "opb", "opc", "opd", "ope"],
    template=f"""
{prefix}
Now this next question:
**Question**: {{question}}
A. {{opa}}
B. {{opb}}
C. {{opc}}
D. {{opd}}
E. {{ope}}

---
**Background Information:** 
Based ONLY on the following context, answer the question strictly with a single letter: 'A', 'B', 'C', 'D' or 'E'.

{{context}}

---

**Answer:**
"""
)


# Benchmark for RAG pipelines
results = []
total = len(df)  # number of samples to evaluate

for i, row in df.iterrows():
    if i >= total:
        break

    start = time.time()

    # Retrieve documents with any of the retreiver variants
    retrieved_docs = bm25_retriever.invoke(row["question"]) # or with any dense retreiver. Check next line
    # retrieved_docs = dense_retriever_gtr.invoke(row["question"]) or with any hybrid retriver. Check next line
    # retrieved_docs = hybrid_with_rerank_sbb(row["question"])
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Create the full prompt string
    full_prompt = prompt.format(
        context=context,
        question=row["question"],
        opa=row["opa"],
        opb=row["opb"],
        opc=row["opc"],
        opd=row["opd"],
        ope=row["ope"]
    )

    # prefill word count (whitespace-delimited)
    prefill_words = len(full_prompt.strip().split())

    # Call the raw pipeline
    output = model_pipeline(full_prompt, max_new_tokens=10, do_sample=False)[0]["generated_text"]

    # Since the model returns the whole prompt + answer, we isolate the new part
    generated_part = output.replace(full_prompt, "").strip().upper()
 
    latency = time.time() - start

    # Process the generated part to extract predicted letter
    match = re.search(r"\b([A-E])\b", generated_part)
    if match:
        predicted = match.group(1)
    else:
        fallback_match = re.search(r"[A-E]", generated_part)
        predicted = fallback_match.group(0) if fallback_match else "?"

    results.append({
        "question": row["question"],
        "expected": row["answer_idx"],
        "predicted": predicted,
        "is_correct": predicted == row["answer_idx"],
        "latency_sec": latency,
        "prefill_words": prefill_words
    })


    if i < 5:
        print(f"\nQuestion: \n{row['question']}")
        print(f"A. {row['opa']}")
        print(f"B. {row['opb']}")
        print(f"C. {row['opc']}")
        print(f"D. {row['opd']}")
        print(f"E. {row['ope']}")
        print(f"\nRetrieved Context:\n{context}")
        print(f"\nModel Answer: {generated_part}")
        print(f"\nExpected: {row['answer_idx']}, Predicted: {predicted}")

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
df_results.to_csv("BM25_llama3_results.csv", index=False)