import warnings, time, re
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from data_preprocessing import load_and_process_medmcqa
from models import login_hf, load_model_and_tokenizer

warnings.filterwarnings('ignore')

# Login
login_hf("hf_token_here")  # Only if gated 

# Load any of the model in 8-bit ("meta-llama/Meta-Llama-3-8B-Instruct", "deepseek-ai/deepseek-llm-7b-chat",  "mistralai/Mistral-7B-Instruct-v0.3")
model_pipeline = load_model_and_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct", "8bit")

# Load data
df = load_and_process_medmcqa()

# Build prompts
def make_prompt(row):
    return (
        "You are a helpful biomedical assistant. Answer the question strictly with a single letter: 'A', 'B', 'C' or 'D'.\n"

        f"Question: {row['question']}\n"
        f"A. {row['opa']}\n"
        f"B. {row['opb']}\n"
        f"C. {row['opc']}\n"
        f"D. {row['opd']}\n"
        "Answer:"
    )

df["prompt"] = df.apply(make_prompt, axis=1)

# Benchmark
results = []
total = len(df) # run on all samples (1001)

for i, (prompt, expected) in enumerate(zip(df["prompt"], df["cop"])):
    if i >= total:
        break

    start = time.time()
    prefill_words = len(prompt.split())
    output = model_pipeline(prompt, max_new_tokens=10, do_sample=False)[0]["generated_text"]
    latency = time.time() - start

    # Extract predicted letter only after 'Answer' line
    after_answer = output.upper().split("ANSWER")[-1]

    match = re.search(r"\b([A-D])\b", after_answer)
    if match:
        predicted = match.group(1)
    else:
        fallback_match = re.search(r"[A-D]", after_answer)
        predicted = fallback_match.group(0) if fallback_match else "?"

    results.append({
        "question": prompt,
        "expected": expected,
        "predicted": predicted,
        "is_correct": predicted == expected,
        "latency_sec": latency,
        "prefill_words": prefill_words
    })

    # print a few samples to verify
    if i < 5:
        print(f"\nPrompt:\n{prompt}")
        print(f"Output:\n{output}")
        print(f"Expected: {expected}, Predicted: {predicted}")

# Compile results dataframe
df_results = pd.DataFrame(results)

# Compute metrics
accuracy = df_results["is_correct"].mean()
avg_latency = df_results["latency_sec"].mean()
avg_prefill_words = df_results["prefill_words"].mean()

# Compute F1 scores
macro_f1 = f1_score(df_results["expected"], df_results["predicted"], average="macro", zero_division=0)
micro_f1 = f1_score(df_results["expected"], df_results["predicted"], average="micro", zero_division=0)
weighted_f1 = f1_score(df_results["expected"], df_results["predicted"], average="weighted", zero_division=0)

# Print summary metrics
print(f"\nMCQ Benchmark completed on {total} samples")
print(f"Accuracy: {accuracy:.2%}")
print(f"Avg latency: {avg_latency:.2f}s")
print(f"Avg prefill words: {avg_prefill_words:.0f}")
print(f"Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")
print("\nClassification report:\n", classification_report(df_results["expected"], df_results["predicted"], zero_division=0))


# Save result to CSV file for further analysis
df_results.to_csv("llm-only_llama3_results.csv", index=False)