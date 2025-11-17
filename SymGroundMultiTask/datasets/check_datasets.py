import pickle


def get_dataset_duplicates(dataset_path):

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    unique = set(dataset)
    duplicate = [item for item in dataset if item not in unique]
    return duplicate


def get_dataset_common_index_pairs(dataset1_path, dataset2_path):

    with open(dataset1_path, "rb") as f:
        dataset1 = pickle.load(f)
    with open(dataset2_path, "rb") as f:
        dataset2 = pickle.load(f)

    # Find common elements
    common_elements = set(dataset1) & set(dataset2)
    
    # Build element-to-index mappings
    index_map1 = {}
    index_map2 = {}

    for i, item in enumerate(dataset1):
        if item in common_elements:
            index_map1.setdefault(item, []).append(i)

    for j, item in enumerate(dataset2):
        if item in common_elements:
            index_map2.setdefault(item, []).append(j)

    # Create index pairs for matching elements
    index_pairs = []
    for elem in common_elements:
        for i in index_map1[elem]:
            for j in index_map2[elem]:
                index_pairs.append((i, j))

    return index_pairs



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset1_path", type=str, default="./datasets/e54/formulas.pkl")
    parser.add_argument("--dataset2_path", type=str, default="./datasets/e54test/formulas.pkl")
    args = parser.parse_args()


    print("dataset1 duplicates:")
    duplicates = get_dataset_duplicates(args.dataset1_path)
    print(duplicates)

    print("dataset2 duplicates:")
    duplicates = get_dataset_duplicates(args.dataset2_path)
    print(duplicates)

    print("indexes of common formulas:")
    indexes = get_dataset_common_index_pairs(
        args.dataset1_path,
        args.dataset2_path
    )
    print(indexes)