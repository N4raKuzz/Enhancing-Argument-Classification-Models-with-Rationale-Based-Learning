import torch
import math
import os
import ast
import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df = pd.read_csv(os.path.abspath("data\snippet.csv"))
#print(df['rationales'][1])
df['rationales'] = df['rationales'].apply(lambda x: ast.literal_eval(x))
rationales = df['rationales'].tolist()
#print(rationales[1])

text = "plot : a separated , glamorous , hollywood couple must pretend to reunite for a press junket of the last movie that they ever shot together . kewl . . . now  i only wish that i could pretend never to have seen this movie  . . . critique :  trite , unfunny , boring and a waste of everyone's talent  . how a premise with such zest and bite can turn into  a movie that doesn't feature any chemistry , any real laughs , any surprises or any spice is beyond me  . how julia roberts is  used solely as a "" puppy dog "" character , puttering around in the background while we endure the complete bitchiness  of zeta-jones' character , who is not  one bit funny or romantic  ( two ideal ingredients in a "" romantic comedy "" ) , is  also beyond me  . and why they chose john cusack , a great , quirky actor in his own right , to play  the most bland , uninteresting and unfetching character ( with zero chemistry with either of his leads ) is further more , beyond me  . and to anybody who decided that this project was "" funny "" enough to greenlight featuring the talents mentioned above , along with billy crystal , christopher walker , seth green and stanley tucci . . . well , what can i say . . . i just don't have the words . so is this the worst movie that i've seen all year ? no . but  it definitely sucks  and it's basically because . . . well ,  it's just not funny  . and for the record ,  allow me to state a few more problems with it  . it  starts off slow , it's got no energy , it doesn't engage you with any of its characters  ( julia barely gets somewhat interesting in the film , everyone else  . . . lame !  ) , it utilizes  way too many  flashbacks to move the story forward , it's  utterly predictable , standard , routine , see-through and uninteresting as a plot and it just sits there on the screen , big and ugly , waiting . . .  waiting for you to laugh or find something in it that is amusing . and then hank azaria shows up . . . aaaaaah , the film's savior ( mind you , some might be offended by his exaggeration of a stereotype , but that's another story altogether ) . but when an experienced "" voice "" actor upstages all of the main stars in a summer "" blockbuster "" romantic comedy with an over-the-top antonio banderas accent , damn dude . . .  your film's in trouble ! !  rent this movie on video just to see  what went wrong  yourself . the references to ricky ricardo and senor wences  ( huh ! ? )  , the  idio-plot points  like when one of the characters goes on the roof to stretch his arms out and relax , but everyone believes that he's going to kill himself ( hardy-har-har ) and  the cheap way  of getting the audience to leave the theater laughing by bringing back a ball-sniffing dog that  has no place being in the location  at the end of the movie , well . . . i could go on . but i won't because i do still respect all of the actors in this film and actually did laugh at azaria , green and tucci's antics from time to time ( ironic , eh . . . what about the leads , dammit ! ) and liked the premise behind the film ( before i saw the finished product , of course ) .  a dud all the way around  . btw , all the talk about this film was that julia roberts was to be in a fat suit for one scene ( her character is supposed to have lost 60 pounds ) , so when the scene finally came , i did get a little excited about what it might look like and then . . . well , it basically just looked like julia roberts in a fat suit !  ugh .  i think i'm gonna start drinking again after this  lame-ass movie  . c'mon hollywood ,  enough with the crud !  where's joblo coming from ? beautiful ( 1/10 ) - my best friend's wedding ( 7/10 ) - notting hill ( 5/10 ) - pretty woman ( 7/10 ) -runaway bride ( 5/10 ) - someone like you ( 4/10 ) - wedding planner ( 3/10 ) - when harry met sally ( 10/10 ) - you've got mail ( 4/10 )"

def create_rationale_mask(rationales, texts):

    mask = [0] * len(texts)

    def matches_rationale(doc_index, rationale):
        if doc_index + len(rationale) > len(texts):
            return False  
        for i, token in enumerate(rationale):
            if texts[doc_index + i] != token:
                return False
        return True

    # Iterate through each token in the document
    for i in range(len(texts)):
        for rationale in rationales:
            if matches_rationale(i, rationale):
                for j in range(len(rationale)):
                    mask[i + j] = 1
    
    return mask

#print(create_rationale_mask(rationales[1], text))
encoded = tokenizer.encode(text)
pad_token_id = tokenizer.tokens_trie("[PAD]") if tokenizer.tokens_trie("[PAD]") is not None else 0
# Truncate
input_ids = encoded.ids[:1024]
# Padding
padding_length = 1024 - len(input_ids)
input_ids += [pad_token_id] * padding_length
input_ids = torch.tensor(input_ids, dtype=torch.int)

print(input_ids)
print(input_ids.shape)