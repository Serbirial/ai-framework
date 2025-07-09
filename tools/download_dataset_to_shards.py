import os
import requests
import pyarrow.parquet as pq

# ------------------------ CONFIG ------------------------

dataset_urls = [
    f"https://huggingface.co/datasets/lemonilia/Elliquiy-Role-Playing-Forums_2023-04/resolve/main/elliquiy-rp_2023-04_0000{i}-of-00006.parquet"
    for i in range(6)
]

save_dir = "./elliquiy_parquet"
shard_output_dir = "./elliquiy_shards"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(shard_output_dir, exist_ok=True)

# ------------------------ DOWNLOAD PARQUET FILES ------------------------

def download_file(url, output_path):
    if os.path.exists(output_path):
        print(f"✔ Already downloaded: {output_path}")
        return
    print(f"⬇ Downloading {os.path.basename(output_path)}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

parquet_files = []
for url in dataset_urls:
    fname = os.path.join(save_dir, url.split("/")[-1])
    download_file(url, fname)
    parquet_files.append(fname)


# ---------------------- FORMAT & SHARD THREADS ------------------------

shard_idx = 0

def format_conversation(category, title, messages):
    lines = [ # category, title, users
        f"<|category|> {category.strip()}",
        f"<|thread-title|> {title.strip()}",
    ]
    for msg in messages:
        speaker = msg.get("from", "Unknown").replace(" ", "_").strip()
        text = msg.get("value", "").strip()
        if len(text) >= 5:
            lines.append(f"<|{speaker}|> {text}")
    data = "\n".join(lines)
    data += "<eos>" # add token to signify end of conversation
    return data

for parquet_path in parquet_files:
    print(f"🔄 Processing {os.path.basename(parquet_path)}...")
    pf = pq.ParquetFile(parquet_path)

    for batch in pf.iter_batches(batch_size=1):
        records = batch.to_pydict()
        for i in range(len(records["messages"])):
            title = records["thread-title"][i]
            messages = records["messages"][i]
            category = records.get("category-name", ["Unknown"])[i]

            if not title or not messages:
                continue

            content = format_conversation(category, title, messages)
            if content.strip():
                shard_path = os.path.join(shard_output_dir, f"thread_{shard_idx:05}.txt")
                with open(shard_path, "w", encoding="utf-8") as fout:
                    fout.write(content.strip())
                shard_idx += 1


print(f"✅ Done. {shard_idx} thread shards saved to: {shard_output_dir}")


"""
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<|category|>", "<|thread-title|>", "<|eos|>"]
})
model.resize_token_embeddings(len(tokenizer))
"""