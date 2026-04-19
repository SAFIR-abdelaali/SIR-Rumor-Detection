import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_test_data_extreme.csv"

def generate_extreme_test_data(n=50):
    data = []
    
    # 1. On utilise le "vocabulaire" que le modele connait (Style PHEME)
    texts_rumors = [
        "BREAKING: Multiple hostages taken inside the cafe! Police surrounding the area.",
        "Unconfirmed reports of a second gunman on the loose. Stay safe!",
        "TERROR ATTACK: Bomb threat reported at the central station, evacuations happening now.",
        "Fake news? People are saying the suspects are already dead.",
        "WARNING: Gunshots heard near the hospital. Do not go outside!"
    ]
    
    texts_real = [
        "Police press conference will be held at 3 PM to discuss the ongoing situation.",
        "Weather update: Heavy rain expected tomorrow across the region.",
        "The prime minister has just released an official statement regarding the event.",
        "Local authorities confirm the area is now secure and roads are reopening.",
        "Tech conference announces its new keynote speaker today."
    ]
    
    for _ in range(n):
        label = np.random.choice([0, 1])
        if label == 1: 
            # PROFIL RUMEUR EXTREME
            text = np.random.choice(texts_rumors)
            followers = np.random.randint(1, 50)         # Compte presque vide (tres suspect)
            verified = 0                                 # Jamais verifie
            retweets = np.random.randint(8000, 25000)    # Hyper viral
            doubt = np.random.randint(20, 60)            # Enormement de gens crient au "Fake"
        else: 
            # PROFIL FIABLE EXTREME
            text = np.random.choice(texts_real)
            followers = np.random.randint(50000, 1000000) # Compte enorme (ex: CNN, BBC)
            verified = 1                                  # Officiel
            retweets = np.random.randint(10, 300)         # Viralite normale
            doubt = np.random.randint(0, 2)               # Personne ne doute
            
        data.append({
            'text': text,
            'followers_count': followers,
            'user_verified': verified,
            'retweet_count': retweets,
            'favorite_count': retweets // 2,
            'reaction_count': retweets // 5,
            'doubt_score': doubt,
            'is_rumor_true_label': label
        })
    
    return pd.DataFrame(data)

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_test_extreme = generate_extreme_test_data(50)
    df_test_extreme.to_csv(OUTPUT_PATH, index=False)
    print(f"Nouveau fichier genere : {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
