import json

input_path = "articles.jsonl"
output_path = "articles_labelled.jsonl"

with open(input_path, "r", encoding="utf-8") as f, \
     open(output_path, "a", encoding="utf-8") as out:

    for content in f:
        data = json.loads(content)         
        print(data["Isi Berita"])

        rating = input("Rating: ")

        out.write(json.dumps({"content": data, "rating": rating}, ensure_ascii=False) + "\n")
        out.flush()