from datasets import load_dataset
import pandas as pd

def load_and_process_medmcqa(dataset_name="openlifescienceai/medmcqa", split="validation", max_samples=None):
    ds = load_dataset(dataset_name, split=split)
    df = pd.DataFrame(ds)

    # Select only top three subject name based on train set
    df = df[df['subject_name'].isin(['Surgery', 'Medicine', 'Pathology'])]
    df = df.reset_index(drop=True)

    # Map numeric answers (0-3) to letters (A-D)
    num_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
    df["cop"] = df["cop"].map(num_to_letter)

    assert df["cop"].notnull().all(), "Unmapped answers found!"

    if max_samples:
        df = df.head(max_samples)

    return df

def load_and_process_medqa_us(dataset_path=r'C:\Users\Student\ragenv\project\medqa-us\test.jsonl', max_samples=None):
    df = pd.read_json(dataset_path, lines=True)

    # Create new columns for each option key
    option_keys = ['A', 'B', 'C', 'D', 'E']

    for key in option_keys:
        col_name = f'op{key.lower()}'
        df[col_name] = df['options'].apply(lambda opts: opts.get(key, None))

    if max_samples:
        df = df.head(max_samples)

    return df

def load_and_process_pubmedqa(dataset_name="qiaojin/PubMedQA", version="pqa_labeled", split="train", max_samples=None):
    ds = load_dataset(dataset_name, version, split=split)
    df = pd.DataFrame(ds)

    if max_samples:
        df = df.head(max_samples)

    return df