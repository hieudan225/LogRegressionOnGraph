import pandas as pd
import numpy as np
import queue
from collections import defaultdict

network_data_path = "./data/epinions/epinions_network.csv"
# network_data_path = "./data/alpha/alpha_network.csv"
network_data = pd.read_csv(network_data_path, names=["buyer", "seller", "rating", "time"])

labels_path = "./data/epinions/epinions_gt.csv"
# labels_path = "./data/alpha/alpha_gt.csv"
labels = pd.read_csv(labels_path, names=["participant", "label"])
# benign_participants = set(labels[labels["label"] == 1]["participant"])
# malicious_participants = set(labels[labels["label"] == -1]["participant"])
labeled_users = labels["participant"]

label_dict = {}

for i, label in labels.iterrows():
    label_dict[label["participant"]] = (int(label["label"]) + 1) / 2
# buyers = network_data['buyer'].unique()
# sellers = network_data['seller'].unique()
# participants = np.unique(np.concatenate([buyers, sellers]))
print("0")
print(network_data.shape[0]) # 13668104
sellers_rating_received = {}
buyers_rating_gave = {}
buyer_degree = defaultdict(set)
seller_degree = defaultdict(set)

for i, txn in network_data.iterrows():
    # print(i)
    buyer = txn["buyer"]
    seller = txn["seller"]
    if buyer not in buyers_rating_gave:
        buyers_rating_gave[txn["buyer"]] = []
    if seller not in sellers_rating_received:
        sellers_rating_received[txn["seller"]] = []
    buyers_rating_gave[txn["buyer"]].append(txn["rating"])
    sellers_rating_received[txn["seller"]].append(txn["rating"])

    buyer_degree[buyer].add(seller)
    seller_degree[seller].add(buyer)
print("1")
buyer_avg_ratings = {}
buyer_std_ratings = {}
seller_avg_ratings = {}
seller_std_ratings = {}


for k, v in buyers_rating_gave.items():
    mean = np.mean(v)
    if len(v) < 2:
        std = 0.00
    else:
        std = np.std(v)
    buyer_avg_ratings[k] = mean
    buyer_std_ratings[k] = std
for k, v in sellers_rating_received.items():
    mean = np.mean(v)
    if len(v) < 2:
        std = 0.00
    else:
        std = np.std(v)
    seller_avg_ratings[k] = mean
    seller_std_ratings[k] = std
print("2")
buyer_rating_top_5_diff_norm = {}
seller_rating_top_5_diff_norm = {}
buyer_rating_top_5_diff = {}
seller_rating_top_5_diff = {}
buyer_rating_last_5_diff = {}
seller_rating_last_5_diff = {}


for _, txn in network_data.iterrows():
    if txn["buyer"] not in labeled_users and txn["seller"] not in labeled_users:
        next
    if txn["buyer"] not in buyer_rating_top_5_diff_norm:
        buyer_rating_top_5_diff_norm[txn["buyer"]] = [0, 0, 0, 0, 0]
        buyer_rating_top_5_diff[txn["buyer"]] = [0, 0, 0, 0, 0]
        buyer_rating_last_5_diff[txn["buyer"]] = [0, 0, 0, 0, 0]
    if txn["seller"] not in seller_rating_top_5_diff_norm:
        seller_rating_top_5_diff_norm[txn["seller"]] = [0, 0, 0, 0, 0]
        seller_rating_top_5_diff[txn["seller"]] = [0, 0, 0, 0, 0]
        seller_rating_last_5_diff[txn["seller"]] = [0, 0, 0, 0, 0]
    buyer_diff = txn["rating"] - seller_avg_ratings[txn["seller"]]
    if buyer_rating_top_5_diff[txn["buyer"]][0] < buyer_diff:
        buyer_rating_top_5_diff[txn["buyer"]][0] = buyer_diff
        buyer_rating_top_5_diff[txn["buyer"]].sort()
    if buyer_rating_last_5_diff[txn["buyer"]][4] > buyer_diff:
        buyer_rating_last_5_diff[txn["buyer"]][4] = buyer_diff
        buyer_rating_last_5_diff[txn["buyer"]].sort()
    buyer_diff_norm = buyer_diff / (seller_std_ratings[txn["seller"]] + 0.001)
    if buyer_rating_top_5_diff_norm[txn["buyer"]][0] < buyer_diff_norm:
        buyer_rating_top_5_diff_norm[txn["buyer"]][0] = buyer_diff_norm
        buyer_rating_top_5_diff_norm[txn["buyer"]].sort()

    seller_diff = txn["rating"] - buyer_avg_ratings[txn["buyer"]]
    if seller_rating_top_5_diff[txn["seller"]][0] < seller_diff:
        seller_rating_top_5_diff[txn["seller"]][0] = seller_diff
        seller_rating_top_5_diff[txn["seller"]].sort()
    if seller_rating_last_5_diff[txn["seller"]][4] > seller_diff:
        seller_rating_last_5_diff[txn["seller"]][4] = seller_diff
        seller_rating_last_5_diff[txn["seller"]].sort()
    seller_diff_norm = seller_diff / (buyer_std_ratings[txn["buyer"]] + 0.001)
    if seller_rating_top_5_diff_norm[txn["seller"]][0] < seller_diff_norm:
        seller_rating_top_5_diff_norm[txn["seller"]][0] = seller_diff_norm
        seller_rating_top_5_diff_norm[txn["seller"]].sort()
print("3")
outdegree = []
indegree = []
avg_ratings_gave = []
avg_ratings_received = []
std_ratings_gave = []
std_ratings_received = []
from_rating_top_1_diff = []
from_rating_last_1_diff = []
from_rating_top_1_diff_norm = []
from_rating_top_2_diff = []
from_rating_last_2_diff = []
from_rating_top_2_diff_norm = []

to_rating_top_1_diff = []
to_rating_last_1_diff = []
to_rating_top_1_diff_norm = []
to_rating_top_2_diff = []
to_rating_last_2_diff = []
to_rating_top_2_diff_norm = []


parsed_label = [] 
parsed_node = []
for node in labeled_users:
    if node in buyer_avg_ratings:
        parsed_node.append(node)
        parsed_label.append(label_dict[node])
        avg_ratings_received.append(0.0)
        std_ratings_received.append(0.0)
        indegree.append(0.0)
        to_rating_top_1_diff.append(0.0)
        to_rating_top_2_diff.append(0.0)

        to_rating_last_1_diff.append(0.0)
        to_rating_last_2_diff.append(0.0)
     

        to_rating_top_1_diff_norm.append(0.0)
        to_rating_top_2_diff_norm.append(0.0)

        avg_ratings_gave.append(buyer_avg_ratings[node])
        std_ratings_gave.append(buyer_std_ratings[node])
        outdegree.append(len(buyer_degree[node]))
        from_rating_top_1_diff.append(buyer_rating_top_5_diff[node][0])
        from_rating_top_2_diff.append(buyer_rating_top_5_diff[node][1])

        from_rating_last_1_diff.append(buyer_rating_last_5_diff[node][0])
        from_rating_last_2_diff.append(buyer_rating_last_5_diff[node][1])

        from_rating_top_1_diff_norm.append(buyer_rating_top_5_diff_norm[node][0])
        from_rating_top_2_diff_norm.append(buyer_rating_top_5_diff_norm[node][1])
    elif node in seller_avg_ratings:
        parsed_node.append(node)
        parsed_label.append(label_dict[node])
        avg_ratings_gave.append(0.0)
        std_ratings_gave.append(0.0)
        outdegree.append(0.0)
        from_rating_top_1_diff.append(0.0)
        from_rating_top_2_diff.append(0.0)

        from_rating_last_1_diff.append(0.0)
        from_rating_last_2_diff.append(0.0)

        from_rating_top_1_diff_norm.append(0.0)
        from_rating_top_2_diff_norm.append(0.0)

        avg_ratings_received.append(seller_avg_ratings[node])
        std_ratings_received.append(seller_std_ratings[node])
        indegree.append(len(seller_degree[node]))
        to_rating_top_1_diff.append(seller_rating_top_5_diff[node][0])
        to_rating_top_2_diff.append(seller_rating_top_5_diff[node][1])

        to_rating_last_1_diff.append(seller_rating_last_5_diff[node][0])
        to_rating_last_2_diff.append(seller_rating_last_5_diff[node][1])

        to_rating_top_1_diff_norm.append(seller_rating_top_5_diff_norm[node][0])
        to_rating_top_2_diff_norm.append(seller_rating_top_5_diff_norm[node][1])
print("4")
dataset = pd.DataFrame({'indegree': indegree, 'outdegree':outdegree,
    'avg_ratings_gave': avg_ratings_gave, "avg_ratings_received": avg_ratings_received,
    'std_ratings_gave': std_ratings_gave, "std_ratings_received": std_ratings_received,
    'from_rating_top_1_diff': from_rating_top_1_diff, 'from_rating_top_2_diff': from_rating_top_2_diff,
    'from_rating_last_1_diff': from_rating_last_1_diff, 'from_rating_last_2_diff': from_rating_last_2_diff,
    'from_rating_top_1_diff_norm': from_rating_top_1_diff_norm, 'from_rating_top_2_diff_norm': from_rating_top_2_diff_norm,
    'to_rating_top_1_diff': to_rating_top_1_diff, 'to_rating_top_2_diff': to_rating_top_2_diff,
    'to_rating_last_1_diff': to_rating_last_1_diff, 'to_rating_last_2_diff': to_rating_last_2_diff,
    'to_rating_top_1_diff_norm': to_rating_top_1_diff_norm, 'to_rating_top_2_diff_norm': to_rating_top_2_diff_norm,
    'label': parsed_label},
    columns=['indegree', 'outdegree', 'avg_ratings_gave', "avg_ratings_received",
    'std_ratings_gave', "std_ratings_received",
    'from_rating_top_1_diff', 'from_rating_top_2_diff',
    'from_rating_last_1_diff', 'from_rating_last_2_diff',
    'from_rating_top_1_diff_norm', 'from_rating_top_2_diff_norm', 
    'to_rating_top_1_diff', 'to_rating_top_2_diff',
    'to_rating_last_1_diff', 'to_rating_last_2_diff',
    'to_rating_top_1_diff_norm', 'to_rating_top_2_diff_norm', 'label'])
dataset = dataset.sample(frac=1)
labels = dataset["label"]
dataset = (dataset-dataset.mean())/dataset.std()
dataset["label"] = labels
dataset.to_csv("dataset.csv", index=False, header=False)
print("done")


