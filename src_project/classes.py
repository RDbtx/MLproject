import json

with open("../Dataset/Labels/train_result.json") as f:
    train_map = json.load(f)
    f.close()
with open("../Dataset/Labels/test_result.json") as f:
    test_map = json.load(f)
    f.close()

print("\ntrain element not found in test:")
for elem in train_map.keys():
    if elem not in test_map.keys():
        print(elem)

print("\ntest element not found in train:")
for elem in test_map.keys():
    if elem not in train_map.keys():
        print(elem)