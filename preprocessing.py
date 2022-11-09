import pandas as pd
import numpy as np

network_data_path = "./data/alpha/alpha_network.csv"
network_data = pd.read_csv(network_data_path, names=["buyer", "seller", "rating", "time"])

labels_path = "./alpha/alpha_gt.csv"
labels = pd.read_csv(labels_path, names=["participant", "label"])
benign_participants = set(labels[labels["label"] == 1]["participant"])
malicious_participants = set(labels[labels["label"] == -1]["participant"])

buyers = network_data['buyer'].unique()
sellers = network_data['seller'].unique()
participants = np.unique(np.concatenate([buyers, sellers]))
def label(participant):
    if participant in benign_participants:
        return 1
    elif participant in malicious_participants:
        return -1
    else:
        return 0
def seller_avg_rating_received(seller):
    sellers = network_data[network_data["seller"] == seller]["rating"]
    if len(sellers) == 0:
        return 0
    return sellers.mean()

def seller_rating_received_std(seller):
    sellers = network_data[network_data["seller"] == seller]["rating"]
    if len(sellers) == 0:
        return 0
    return sellers.std()
vectorized_label = np.vectorize(label)
vectorized_seller_avg_rating_received = np.vectorize(seller_avg_rating_received)
vectorized_seller_rating_received_std = np.vectorize(seller_rating_received_std)
expanded_labels = vectorized_label(participants)
seller_labels = vectorized_label(network_data["seller"])
seller_avg_ratings = vectorized_seller_avg_rating_received(network_data["seller"])
seller_ratings_std = vectorized_seller_rating_received_std(network_data["seller"])

network_data["seller_avg_rating"] = seller_avg_ratings
network_data["seller_ratings_std"] = seller_ratings_std
network_data["seller_label"] = seller_labels

def average_ratings_a_buyer_gave(buyer):
    rating_gave = network_data[network_data["buyer"] == buyer]["rating"]
    return rating_gave.mean() if len(rating_gave) != 0 else 0
def ratings_a_buyer_gave_std(buyer):
    rating_gave = network_data[network_data["buyer"] == buyer]["rating"]
    return rating_gave.std() if len(rating_gave) != 0 else 0
vectorized_average_ratings_a_buyer_gave = np.vectorize(average_ratings_a_buyer_gave)
avg_ratings_gave = vectorized_average_ratings_a_buyer_gave(network_data["buyer"])
vectorized_ratings_a_buyer_gave_std = np.vectorize(ratings_a_buyer_gave_std)
ratings_gave_std = vectorized_ratings_a_buyer_gave_std(network_data["buyer"])

network_data["buyer_avg_rating"] = avg_ratings_gave
network_data["buyer_ratings_std"] = ratings_gave_std

def count_outdegree(buyer):
    return len(network_data[network_data["buyer"] == buyer])
def count_indegree(seller):
    return len(network_data[network_data["seller"] == seller])
vectorized_count_indegree = np.vectorize(count_indegree)
vectorized_count_outdegree = np.vectorize(count_outdegree)
indegree = vectorized_count_indegree(participants)
outdegree = vectorized_count_outdegree(participants)

network_data["abs_diff_buyer_gave"] = network_data["rating"] - network_data["seller_avg_rating"]
network_data["abs_diff_seller_receive"] = network_data["rating"] - network_data["buyer_avg_rating"]
network_data["rel_diff_buyer_gave"] = network_data["abs_diff_buyer_gave"].abs() / network_data["seller_ratings_std"]
network_data["rel_diff_seller_receive"] = network_data["abs_diff_seller_receive"].abs() / network_data["buyer_ratings_std"]

def top_k_diff_buyer(buyer, k):
	abs_diff_buyer_gave = network_data[network_data["buyer"] == buyer]["abs_diff_buyer_gave"]
	return abs_diff_buyer_gave.nlargest(5).iloc[k] if len(abs_diff_buyer_gave) > k else 0

vectorized_top_k_diff_buyer = np.vectorize(top_k_diff_buyer)

top_1_diff_buyer = vectorized_top_k_diff_buyer(participants, 0)
top_2_diff_buyer = vectorized_top_k_diff_buyer(participants, 1)
top_3_diff_buyer = vectorized_top_k_diff_buyer(participants, 2)
top_4_diff_buyer = vectorized_top_k_diff_buyer(participants, 3)
top_5_diff_buyer = vectorized_top_k_diff_buyer(participants, 4)

def top_k_diff_seller(seller, k):
	abs_diff_seller_receive = network_data[network_data["seller"] == seller]["abs_diff_seller_receive"]
	return abs_diff_seller_receive.nlargest(5).iloc[k] if len(abs_diff_seller_receive) > k else 0

vectorized_top_k_diff_seller = np.vectorize(top_k_diff_seller)

top_1_diff_seller = vectorized_top_k_diff_seller(participants, 0)
top_2_diff_seller = vectorized_top_k_diff_seller(participants, 1)
top_3_diff_seller = vectorized_top_k_diff_seller(participants, 2)
top_4_diff_seller = vectorized_top_k_diff_seller(participants, 3)
top_5_diff_seller = vectorized_top_k_diff_seller(participants, 4)


def last_k_diff_buyer(buyer, k):
	abs_diff_buyer_gave = network_data[network_data["buyer"] == buyer]["abs_diff_buyer_gave"]
	return abs_diff_buyer_gave.nsmallest(5).iloc[k] if len(abs_diff_buyer_gave) > k else 0

vectorized_last_k_diff_buyer = np.vectorize(last_k_diff_buyer)

last_1_diff_buyer = vectorized_last_k_diff_buyer(participants, 0)
last_2_diff_buyer = vectorized_last_k_diff_buyer(participants, 1)
last_3_diff_buyer = vectorized_last_k_diff_buyer(participants, 2)
last_4_diff_buyer = vectorized_last_k_diff_buyer(participants, 3)
last_5_diff_buyer = vectorized_last_k_diff_buyer(participants, 4)

def last_k_diff_seller(seller, k):
	abs_diff_seller_receive = network_data[network_data["seller"] == seller]["abs_diff_seller_receive"]
	return abs_diff_seller_receive.nsmallest(5).iloc[k] if len(abs_diff_seller_receive) > k else 0

vectorized_last_k_diff_seller = np.vectorize(last_k_diff_seller)

last_1_diff_seller = vectorized_last_k_diff_seller(participants, 0)
last_2_diff_seller = vectorized_last_k_diff_seller(participants, 1)
last_3_diff_seller = vectorized_last_k_diff_seller(participants, 2)
last_4_diff_seller = vectorized_last_k_diff_seller(participants, 3)
last_5_diff_seller = vectorized_last_k_diff_seller(participants, 4)

def top_k_rel_diff_buyer(buyer, k):
	rel_diff_buyer_gave = network_data[network_data["buyer"] == buyer]["rel_diff_buyer_gave"]
	return rel_diff_buyer_gave.nlargest(5).iloc[k] if len(rel_diff_buyer_gave) > k else 0

vectorized_top_k_rel_diff_buyer = np.vectorize(top_k_rel_diff_buyer)

top_1_rel_diff_buyer = vectorized_top_k_rel_diff_buyer(participants, 0)
top_2_rel_diff_buyer = vectorized_top_k_rel_diff_buyer(participants, 1)
top_3_rel_diff_buyer = vectorized_top_k_rel_diff_buyer(participants, 2)
top_4_rel_diff_buyer = vectorized_top_k_rel_diff_buyer(participants, 3)
top_5_rel_diff_buyer = vectorized_top_k_rel_diff_buyer(participants, 4)

def top_k_rel_diff_seller(seller, k):
	rel_diff_seller_receive = network_data[network_data["seller"] == seller]["rel_diff_seller_receive"]
	return rel_diff_seller_receive.nlargest(5).iloc[k] if len(rel_diff_seller_receive) > k else 0

vectorized_top_k_rel_diff_seller = np.vectorize(top_k_rel_diff_seller)

top_1_rel_diff_seller = vectorized_top_k_rel_diff_seller(participants, 0)
top_2_rel_diff_seller = vectorized_top_k_rel_diff_seller(participants, 1)
top_3_rel_diff_seller = vectorized_top_k_rel_diff_seller(participants, 2)
top_4_rel_diff_seller = vectorized_top_k_rel_diff_seller(participants, 3)
top_5_rel_diff_seller = vectorized_top_k_rel_diff_seller(participants, 4)

dataset = pd.DataFrame({'node': participants, 'indegree': indegree, 'outdegree':outdegree,
	'avg_ratings_gave': avg_ratings_gave , 'top_1_diff_buyer': top_1_diff_buyer, 'top_2_diff_buyer': top_2_diff_buyer, 
	'top_3_diff_buyer': top_3_diff_buyer, 'top_4_diff_buyer': top_4_diff_buyer, 'top_5_diff_buyer': top_5_diff_buyer,
	'top_1_diff_seller': top_1_diff_seller, 'top_2_diff_seller': top_2_diff_seller, 
	'top_3_diff_seller': top_3_diff_seller, 'top_4_diff_seller': top_4_diff_seller, 'top_5_diff_seller': top_5_diff_seller,
	'last_1_diff_buyer': last_1_diff_buyer, 'last_2_diff_buyer': last_2_diff_buyer, 
	'last_3_diff_buyer': last_3_diff_buyer, 'last_4_diff_buyer': last_4_diff_buyer, 'last_5_diff_buyer': last_5_diff_buyer,
	'last_1_diff_seller': last_1_diff_seller, 'last_2_diff_seller': last_2_diff_seller, 
	'last_3_diff_seller': last_3_diff_seller, 'last_4_diff_seller': last_4_diff_seller, 'last_5_diff_seller': last_5_diff_seller,
	'top_1_rel_diff_buyer': top_1_rel_diff_buyer, 'top_2_rel_diff_buyer': top_2_rel_diff_buyer, 
	'top_3_rel_diff_buyer': top_3_rel_diff_buyer, 'top_4_rel_diff_buyer': top_4_rel_diff_buyer, 'top_5_rel_diff_buyer': top_5_rel_diff_buyer,
	'top_1_rel_diff_seller': top_1_rel_diff_seller, 'top_2_rel_diff_seller': top_2_rel_diff_seller, 
	'top_3_rel_diff_seller': top_3_rel_diff_seller, 'top_4_rel_diff_seller': top_4_rel_diff_seller, 'top_5_rel_diff_seller': top_5_rel_diff_seller,
	'label': expanded_labels}, 
	columns=['node', 'indegree', 'outdegree', 'top_1_diff_buyer', 'top_2_diff_buyer', 'top_3_diff_buyer', 'top_4_diff_buyer', 
	'top_5_diff_buyer', 'top_1_diff_seller', 'top_2_diff_seller', 'top_3_diff_seller', 'top_4_diff_seller', 'top_5_diff_seller',
	'last_1_diff_buyer', 'last_2_diff_buyer', 'last_3_diff_buyer', 'last_4_diff_buyer', 
	'last_5_diff_buyer', 'last_1_diff_seller', 'last_2_diff_seller', 'last_3_diff_seller', 'last_4_diff_seller', 'last_5_diff_seller',
	'top_1_rel_diff_buyer', 'top_2_rel_diff_buyer', 'top_3_rel_diff_buyer', 'top_4_rel_diff_buyer', 
	'top_5_rel_diff_buyer', 'top_1_rel_diff_seller', 'top_2_rel_diff_seller', 'top_3_rel_diff_seller', 'top_4_rel_diff_seller', 'top_5_rel_diff_seller', 'label'])

dataset = dataset.loc[dataset['label'] != 0]

dataset.to_csv("dataset.csv")



