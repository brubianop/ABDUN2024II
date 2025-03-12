import pandas as pd
import requests
import random
import time

def get_rvw(game_id, prmtrs):
    
    slp = f"https://store.steampowered.com/appreviews/{game_id}?json=1"
    time.sleep(2) #bot
    try:
        res = requests.get(url = slp, params = prmtrs)
    except:
        print(f"Error.")
        time.sleep(2)
        return get_rvw(game_id, prmtrs)

    try:
        rvw_batch = res.json()
    except:
        print(f"json req error.")
        time.sleep(2)
        return get_rvw(game_id, prmtrs)

    return rvw_batch


def parse_rvw_batch(game_id, rvw_batch):
    
    r_batch = []
    for rvw in rvw_batch["reviews"]:
        review = {
            "game_id": game_id,
            "recommendation_id": rvw["recommendationid"],
            "steam_id": rvw["author"]["steamid"],
            
            "review": rvw["review"],
            "timestamp_created": rvw["timestamp_created"],
            "timestamp_updated": rvw["timestamp_updated"],
            "voted_up": rvw["voted_up"],
            "votes_up": rvw["votes_up"],
            "weighted_vote_score": rvw["weighted_vote_score"],
            "steam_purchase": rvw["steam_purchase"],
            "received_for_free": rvw["received_for_free"],
            "written_during_early_access": rvw["written_during_early_access"]
            }

        r_batch.append(review)
        
    return r_batch


def main():

    with open("g_id.txt", 'r') as f:
        game_b = [l.strip() for l in f]
    
    prmtrs = {
        "json": 1,
        "cursor": '*',
        "num_per_page": 100,
        'filter': "all",
        "language": "english",
        "review_type": "all",
        "purchase_type": "all",
        "day_range": 9223372036854775807
        }

    MAX_RVW = 30000
    
    print(game_b)
    for game_id in game_b:

        cursor = False
        cursors = []
        reviews = []
        cur_rvw = 0
        print(f"Current id: {game_id}")
        while cursor not in cursors and cur_rvw <= MAX_RVW:
            cursors.append(cursor) #Current cursor
            
            rvw_b = get_rvw(game_id, prmtrs)
            
            reviews += parse_rvw_batch(game_id, rvw_b)
            cur_rvw = len(reviews)
            print(f"Current reviews: {cur_rvw}")
            
            cursor = rvw_b["cursor"] #Cursor updated
            prmtrs['cursor'] = cursor
            
        prmtrs["cursor"] = '*'
        pd.DataFrame(reviews).to_csv(f"sreviews_{game_id}.csv")


if __name__ == "__main__":
    main()
