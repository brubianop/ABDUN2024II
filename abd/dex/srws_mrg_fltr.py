import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("all")

MAX_R = 10000
NGRAM_SZ = 10

with open("wdata/stop_words_english.txt", 'r') as f:
    stw = [l.strip() for l in f]
    


#SEC I & II
def word_freq(word_l):
    
    gw_freq = {}
    wpr_freql = []
    g_wl = sum(word_l, [])
    
    for w in g_wl:
        if w not in gw_freq:
            gw_freq[w] = 1
        else:
            gw_freq[w] += 1

    g_sfreq = {k: v for k, v in sorted(gw_freq.items(),
                                       key = lambda val: val[1],
                                       reverse = True)}
    
    plt.barh(y = list(g_sfreq.keys())[0:25], 
             width = list(g_sfreq.values())[0:25], 
             height = 0.2)
    plt.title("Top 25 Words")
    plt.gca().invert_yaxis()
    plt.show()
    
    #plt.barh(y = list(g_sfreq.keys())[0:-25], 
    #         width = [abs(x) for x in list(g_sfreq.values())][0:-25], 
    #         height = 0.2)
    #plt.title("Top 25 Negative")
    #plt.gca().invert_yaxis()
    #plt.show()

    for wl in word_l:
        wpr_freq = {}
        for w in wl:
            if w not in wpr_freq:
                wpr_freq[w] = 1
            else:
                wpr_freq[w] += 1
        wpr_freql.append(wpr_freq)
        
    return g_sfreq, wpr_freql


#SEC I 

def cos_sim(gen_freq, rvw_freql):
    
    df = pd.DataFrame(rvw_freql, columns = list(gen_freq)[0:15]).fillna(0)
    data_m = df.values.tolist()
    
    c_sim_m = np.empty((len(data_m), len(data_m)))
    prsn_cor_m = np.empty((len(data_m), len(data_m)))
    for i in range(len(data_m)):
        for j in range(len(data_m)):
            #cosine
            dot = sum(a*b for a, b in zip(data_m[i], data_m[j]))
            p_norm = np.linalg.norm(data_m[i])*np.linalg.norm(data_m[j])
            if p_norm != 0:
                c_sim_m[i, j] = round(float(dot) / p_norm, 4)

            
    print(c_sim_m)
    
    plt.plot(figsize = (8, 8))
    sns.heatmap(c_sim_m, cmap = "magma")
    plt.title("Matriz Similaridad Coseno reviews")
    plt.show()
    

def pearson_corr(gen_freq, rvw_freql):
    df = pd.DataFrame(rvw_freql, columns = list(gen_freq)[0:25]).fillna(0)
    corr_m = df.corr(method = "pearson")
    
    print(corr_m)
    plt.plot(figsize = (8, 6))
    sns.heatmap(corr_m, xticklabels = 1, yticklabels = 1, cmap = "viridis")
    plt.title("Matriz de correlación")
    plt.show()

#END SEC I 


#SEC II

def NRC_emolex_words(gen_freq):
    
    rNRC_lex_df = pd.read_csv("wdata/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt",
                          names = ["word", "emotion", "association"],
                          skiprows = 45,
                          sep = "\t",
                          keep_default_na = False)
    NRC_lex_df = rNRC_lex_df.pivot(index='word', 
                                   columns='emotion', 
                                   values='association').reset_index()
    pos_w = NRC_lex_df[NRC_lex_df.positive == 1].word.tolist()
    neg_w = NRC_lex_df[NRC_lex_df.negative == 1].word.tolist()
    
    
    con_w = {"positive_w": 0, "negative_w": 0}
    #con = {"positive_count": 0, "negative_count": 0}
    for w in list(gen_freq.keys()):
        if w in pos_w and w not in neg_w:
            con_w["positive_w"] += gen_freq[w]
            #con["positive_count"] += 1
        elif w in neg_w and w not in pos_w:
            con_w["negative_w"] += gen_freq[w]
            #con["negative_count"] += 1
    
    print(f"Positive_w - Negative_w vals: {list(con_w.values())}")
    
    
    plt.bar(x = list(con_w.keys()), 
            height = list(con_w.values()),
            color = ['#1b9e77', '#fdaa48'])
    plt.title("Positive - Negative Score")
    plt.show()
    
    
    #plt.bar(x = list(con.keys()), height = list(con.values()))
    #plt.show()
    

#END SEC II


#SEC III
def bt_grams(bigrams, trigrams):
    
    b_dct = {}
    prsd = 0
    for bgrams in bigrams:
        prsd = dict(nltk.FreqDist(bgrams))
        for k in prsd.keys():
            if str(k) not in b_dct:
                b_dct[str(k)] = prsd[k]
            else:
                b_dct[str(k)] += prsd[k]

    

    t_dct = {}
    for tgrams in trigrams:
        prsd = dict(nltk.FreqDist(tgrams))
        for k in prsd.keys():
            if str(k) not in t_dct:
                t_dct[str(k)] = prsd[k]
            else:
                t_dct[str(k)] += prsd[k]

    #sort
    b_dct_sorted = {k: v for k, v in sorted(b_dct.items(),
                                            key = lambda val: val[1],
                                            reverse = True)}
    
    t_dct_sorted = {k: v for k, v in sorted(t_dct.items(),
                                            key = lambda val: val[1],
                                            reverse = True)}
    print(f"BIGRAMS SORTED\n{b_dct_sorted}\n")
    plt.barh(y = list(b_dct_sorted.keys())[0:25], 
             width = list(b_dct_sorted.values())[0:25], 
             height = 0.2)
    plt.title("Top 25 bigrams")
    plt.gca().invert_yaxis()
    plt.show()
    
    plt.barh(y = list(t_dct_sorted.keys())[0:25], 
             width = list(t_dct_sorted.values())[0:25], 
             height = 0.2)
    plt.title("Top 25 trigrams")
    plt.gca().invert_yaxis()
    plt.show()
    
    #bgrams and tgrams soted by frequency. Execute 
    #print(f"BIGRAMS SORTED\n{b_dct_sorted}\n")
    
    #print(f"TRIGRAMS SORTED\n{t_dct_sorted}\n")

#END SEC III

#SEC IV
def NRC_VAD_review(review_r, review_l):
    
    NRC_VAD_lex = dict(map(lambda l: (l[0], float(l[1])), 
                     [ line.split('\t') for line in open("wdata/NRC-VAD-Lexicon.txt") ]))
    
    rrate_l = []
    nrate_l = []
    for r in review_l:
        ovl_val = 0
        for w in r:
            if w in NRC_VAD_lex:
                ovl_val += NRC_VAD_lex[w]
        
        if len(r):
            ovl_val /= len(r)
        
        if ovl_val >= 0.1:
            rrate_l.append(1)
        elif ovl_val <= -0.1:
            rrate_l.append(-1)
        else:
            rrate_l.append(0)
        nrate_l.append(ovl_val)
    
    review_df = pd.DataFrame({"review": review_r,
                              "rating": rrate_l,
                              "numeric rating": nrate_l})
    
    norm = 1 / review_df.shape[0]

    plt.bar(x = ["pos", "neg", "neu"], 
            height = [(review_df["rating"] == 1).sum() * norm,
                      (review_df["rating"] == -1).sum() * norm,
                      (review_df["rating"] == 0).sum() * norm],
            color = ['#1b9e77', '#fdaa48', '#A890F0'])
    
    plt.title("REVIEW")
    plt.show()
    
    return {"pos": (review_df["rating"] == 1).sum() * norm,
            "neg": (review_df["rating"] == -1).sum() * norm,
            "neu": (review_df["rating"] == 0).sum() * norm}


def NRC_VAD_comp_VADER(NRC_based, VADER_based):
    print(F"NRC: {NRC_based}")
    plt.bar(x = ["pos", "neg", "neu"], 
            height = [NRC_based["pos"],
                      NRC_based["neg"],
                      NRC_based["neu"]],
            color = ['#1b9e77', '#fdaa48', '#A890F0'])
    
    plt.title("NRC BASED")
    plt.show()
    
    
    print(F"VADER: {VADER_based}")
    plt.bar(x = ["pos", "neg", "neu"], 
            height = [VADER_based["pos"],
                      VADER_based["neg"],
                      VADER_based["neu"]],
            color = ['#1b9e77', '#fdaa48', '#A890F0'])
    plt.title("VADER BASED")
    plt.show()
    
    

def rvws_fltr(srvws_f):
    
    srvws_f = glob.glob("sreviews_*.csv")
    for ds in srvws_f:
        print(f"Game: {ds}")
        srvws_df = pd.read_csv(ds)
        bgrams = []
        tgrams = []
        flwl = []
        vdr_l = []
        #r = REVIEW
        for i, r in enumerate(srvws_df["review"]):
            if i == MAX_R:
                break
            
            #NLTK STOP WORDS
            nltk_stw = nltk.corpus.stopwords.words("english")
            nltk_stw.remove("not")
            
            #Naive Analysis
            
            w = [wrd.lower() for wrd in str(r).split()] #raw
            
            flwl.append([wr for wr in w 
                                 if wr not in stw 
                                 and wr not in nltk_stw
                                 and wr.isalpha()]) #filter
            
            
            
            #BIGRAMS - TRIGRAMS CONSTRUCT
            
            nlp_w = [wrds.lower() for wrds in nltk.word_tokenize(str(r)) 
                                     if wrds.isalpha() 
                                     and wrds.lower() not in nltk_stw
                                     and wrds.lower() not in stw]
            
            #filtered_r = ' '.join(nlp_w) #filtered review
            bgrams.append(nltk.ngrams(nlp_w, 2))
            tgrams.append(nltk.ngrams(nlp_w, 3))
            
            
            #VADER
            v_score = {"review": r, "rating": 0}
            
            scr_c = 0
            
            
            rsentences = nltk.sent_tokenize(str(r))
            
            for rsent in rsentences:
                sia = SentimentIntensityAnalyzer()
                scr  = sia.polarity_scores(rsent)
                scr_c += scr["compound"]
                
            if len(rsentences):   
                scr_c /= len(rsentences)
            
            if scr_c >= 0.1:
                v_score["rating"] = 1
            elif scr_c <= -0.1:
                v_score["rating"] = -1
            else:
                v_score["rating"] = 0
            
            vdr_l.append(v_score)
            
                
                
                
                
            
            
            
            
            
        
        #COMMENTS NAIVE FREQ 
        gen_freq, rvw_freql = word_freq(flwl)
        
        
        #SEC I NAIVE (relations)
        #cos_sim(gen_freq, rvw_freql)
        pearson_corr(gen_freq, rvw_freql)
        
        
        #SEC II Freq and (NRC_EMOLEX) Lexicon Analysis
        NRC_emolex_words(gen_freq)
        
        
        #SEC III NGRAMS FREQ (Future work)
        bt_grams(bgrams, tgrams)
        
        
        #SEC IV NLP ANALYSIS (Comparison VADER)
        
        #NRC_VALENCE DATA MODEL (Per review)
        rvw_score = NRC_VAD_review(srvws_df["review"][:MAX_R], flwl)
        
        #VADER PRE TRAINED MODEL SCORE
        pos_r = neg_r = neu_r = 0
        for rs in vdr_l:
            if rs["rating"] == 1:
                pos_r += 1
            elif rs["rating"] == -1:
                neg_r += 1
            else:
                neu_r  += 1
        
        norm = 1 / len(vdr_l)
        vdr_r = {"pos": pos_r * norm, 
                 "neg": neg_r * norm, 
                 "neu": neu_r * norm}
        
        NRC_VAD_comp_VADER(rvw_score, vdr_r)
        
        
        


def main():
    #works
    #srvws_df = pd.concat([pd.read_csv(fl) for fl in srvws_f], ignore_index = True)
    srvws_f = glob.glob("sreviews_*.csv")
    rvws_fltr(srvws_f)

    
if __name__ == "__main__":
    main()
