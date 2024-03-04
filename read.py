import json


if __name__ == "__main__":
    file = "data/train.json"
    with open(file) as f:
        raw_data = f.read()
    data = json.loads(raw_data)

    for record in data:
        print(record.keys())
        print(f"*** {record.get('filename')}")
        table = record.get("table_ori")
        print(table[0])
        print(table[1])
