# import json
#
# from FlagEmbedding import BGEM3FlagModel
#
# model = BGEM3FlagModel('BAAI/bge-m3',  use_fp16=True)
#
# with open("joined_reviews.json") as fp:
#     data = json.load(fp)
#
#
# for review in data:
#     categories = ""
#     for i in range(len(review['category'])):
#         if i==0:
#             categories += f"{review['category'][i]}"
#         else:
#             categories += f"/{review['category'][i]}"
#     business_info = f"Customer review for a {categories}"
#     print(business_info)
#     print(review)
#     print(model.compute_score([review['text'],business_info], # a smaller max length leads to a lower latency
#                               weights_for_different_modes=[0.4, 0.2, 0.4]))

import pandas as pd
from FlagEmbedding import BGEM3FlagModel

# Initialize model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Load CSV
data = pd.read_csv("google_reviews_labeled_combined_with_json.csv")

# If 'category' column is stored as a string of list, convert it to actual list
# Example: "['Boutique', 'Clothing']" -> ['Boutique', 'Clothing']
import ast

data['categories'] = data['categories'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Iterate through rows
for _, review in data.iterrows():
    categories = "/".join(review['categories'])  # join categories with slash
    business_info = f"Customer review for a {categories}"

    print(business_info)
    print(review)

    score = model.compute_score(
        [review['text'], business_info],
        weights_for_different_modes=[0.4, 0.2, 0.4]
    )

    print(score)
