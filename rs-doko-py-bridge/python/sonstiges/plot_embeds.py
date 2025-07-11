import numpy as np
from scipy.spatial.distance import cosine

card_weights = [[ 4.5122e-3, -1.9423e-3, -3.4847e-3, -1.8674e-3,  5.7003e-3,  6.0742e-3,
                  7.9234e-3,  1.2065e-3],
                [ 4.2453e-1,  5.8341e-1,  2.7753e-1, -7.9700e-2, -1.8982e-2,  6.0395e-1,
                  -6.7470e-1,  4.9543e-1],
                [-1.9410e-1,  2.9728e-1,  7.1135e-1,  5.4439e-2, -2.6640e-1,  -1.2266e0,
                 6.8808e-2,  3.5453e-1],
                [-2.0616e-2,  2.9280e-1,  1.3549e-1,  7.6853e-2,  -1.0412e0,  4.1288e-1,
                 -5.8936e-2,  1.1528e-1],
                [ 2.5524e-1,  9.1617e-1, -8.4594e-2,  4.3402e-2, -3.4900e-2,  2.3158e-2,
                  -7.2922e-1, -1.0427e-1],
                [ 2.2232e-1,  1.5870e-1, -3.1442e-2,  3.9484e-2,  5.9419e-2,  8.3564e-2,
                  1.0149e-1,  9.6860e-1],
                [-2.9560e-1, -3.2940e-2, -3.9801e-1, -9.2009e-2, -2.7594e-1, -9.5805e-1,
                 9.9405e-2,  4.7805e-1],
                [ 9.4075e-2, -1.6615e-3,  5.2236e-2,  1.5808e-1,   1.4321e0,  9.6995e-1,
                  -1.1940e-1, -7.6262e-2],
                [-1.3476e-1,  3.1490e-1,  -2.3542e0,  3.7482e-1, -1.0416e-1, -7.0083e-2,
                 -2.6643e-2, -5.3610e-2],
                [ 4.2239e-1,  6.8457e-1,  1.2792e-1, -6.4748e-1, -1.8636e-1,  1.3399e-1,
                  2.1629e-1,  2.7729e-1],
                [-2.5810e-1,   1.0963e0, -4.0725e-1,  1.2870e-1, -1.6340e-1, -1.2876e-1,
                 -3.8126e-2, -5.3657e-1],
                [ 4.7295e-2,  6.1963e-3,   1.0234e0, -7.2260e-4,   1.0829e0, -9.0468e-3,
                  1.8995e-2,  6.4177e-2],
                [-1.9461e-1, -3.2922e-1, -2.8633e-1, -2.7596e-1,   1.3979e0, -9.1676e-1,
                 2.2973e-1,  5.3075e-2],
                [ 8.4289e-1, -7.0567e-1,  7.3472e-3,  3.2231e-1,  6.0088e-2,  9.3527e-1,
                  -2.2959e-1, -6.4026e-1],
                [ 8.7288e-1,  -1.4275e0, -3.8232e-2,  3.1509e-2, -1.1189e-1, -6.0024e-1,
                  -2.6110e-1,  2.0999e-1],
                [-2.6239e-1,  6.9924e-1, -1.8745e-1,  1.0074e-1,  1.4386e-1,  3.9033e-1,
                 -1.0639e-1,  1.7107e-1],
                [ 3.9190e-2,  1.1157e-1,  2.0973e-2,   5.9599e0, -2.7454e-2,  1.8166e-2,
                  -2.7352e-4,  6.3681e-3],
                [ 8.6726e-1, -8.3729e-1,  2.0453e-1, -3.3799e-1, -1.6268e-1,  1.7907e-1,
                  9.1525e-1, -4.1157e-1],
                [ 4.9909e-1, -3.4665e-1, -9.8540e-2, -2.4747e-1, -2.2464e-2,  -1.0712e0,
                  -3.3148e-1,  -1.3811e0],
                [-6.0121e-1, -2.4249e-1,  1.0755e-1, -8.9732e-1, -1.1070e-1,   1.1158e0,
                 7.1856e-2, -1.7674e-1],
                [ -1.0694e0, -9.6557e-1,  5.0551e-3, -1.9013e-1,  1.7872e-2, -8.7066e-2,
                  -1.0405e0,  1.4164e-1],
                [-4.6000e-2,  6.0582e-1,  2.4193e-2,  1.2640e-1, -3.9423e-2,  1.4649e-1,
                 8.6452e-1,  8.8006e-3],
                [ 6.3636e-1,   1.0388e0, -6.6998e-1,  1.1154e-1,  1.6419e-1, -1.3142e-1,
                  2.0365e-1,  1.4979e-1],
                [-9.8000e-1, -8.7221e-1, -9.4830e-3,  1.4596e-1, -9.2385e-2,  6.7444e-1,
                 5.5864e-1,  4.8089e-1],
                [ -1.4859e0, -2.6105e-1, -3.2151e-2,  2.6971e-2, -8.1440e-2, -5.2526e-1,
                  2.6840e-1,  -1.0284e0]]



mapping = {
    0: "None",
    1: "♦ 9",
    2: "♦ 10",
    3: "♦ J",
    4: "♦ Q",
    5: "♦ K",
    6: "♦ A",
    7: "♥ 9",
    8: "♥ 10",
    9: "♥ J",
    10: "♥ Q",
    11: "♥ K",
    12: "♥ A",
    13: "♣ 9",
    14: "♣ 10",
    15: "♣ J",
    16: "♣ Q",
    17: "♣ K",
    18: "♣ A",
    19: "♠ 9",
    20: "♠ 10",
    21: "♠ J",
    22: "♠ Q",
    23: "♠ K",
    24: "♠ A"
}

card_weights = np.array(card_weights)


# Rechne Pik Ass - Pik 9 + Kreuz 9
pik_ass_vector = card_weights[list(mapping.keys())[list(mapping.values()).index("♠ A")]]
pik_9_vector = card_weights[list(mapping.keys())[list(mapping.values()).index("♠ 9")]]
kreuz_9_vector = card_weights[list(mapping.keys())[list(mapping.values()).index("♣ 9")]]
herz_9_vector = card_weights[list(mapping.keys())[list(mapping.values()).index("♥ 9")]]
karo_9_vector = card_weights[list(mapping.keys())[list(mapping.values()).index("♦ 9")]]

result_vector = pik_ass_vector - pik_9_vector + herz_9_vector

# Berechne die Distanzen zu allen anderen Karten
distances = np.linalg.norm(card_weights - result_vector, axis=1)

# Finde den Index des nächsten Vektors
closest_index = np.argmin(distances)

# Gib den nächsten Vektor aus
print(f"Nächster Vektor: {mapping[closest_index]}")



for i, v in enumerate(card_weights):
    # Calculate cosine distances to all other cards
    distances = [cosine(v, other) for other in card_weights]
    # Get indices of the 4 closest cards (excluding the card itself)
    closest_indices = np.argsort(distances)[1:5]
    closest_cards = [mapping[idx] for idx in closest_indices]

    print(f"{mapping[i]}: {', '.join(closest_cards)}")

print("=====================================")

for i, v in enumerate(card_weights):
    # Calculate distances to all other cards
    distances = np.linalg.norm(card_weights - v, axis=1)
    # Get indices of the 4 closest cards (excluding the card itself)
    closest_indices = np.argsort(distances)[1:5]
    closest_cards = [mapping[idx] for idx in closest_indices]

    print(f"{mapping[i]}: {', '.join(closest_cards)}")

import plotly.express as px




x = []
y = []
z = []
texts = []

for i, v in enumerate(card_weights):
    x.append(v[0])
    y.append(v[1])
    z.append(v[2])
    texts.append(mapping[i])

fig = px.scatter_3d(
 x=x, y=y, z=z,
 text=texts,
)

fig.show()


# player_embeds = [[ 0.9549, -1.8551],
#  [-1.2852,  0.0169],
#  [-0.0277, -1.6517],
#  [ 1.2446,  0.0288],
#  [-0.0303,  1.7337]]
#
#
# import plotly.express as px
#
# x = []
# y = []
# texts = []
#
# for i, v in enumerate(player_embeds):
#     x.append(v[0])
#     y.append(v[1])
#     texts.append(i)
#
#     fig = px.scatter(
#         x=x, y=y,
#         text=texts,
#     )
#
# fig.show()