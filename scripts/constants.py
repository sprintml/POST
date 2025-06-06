
TEXT_LABEL_DICT={
    "sst2": [[" terrible"," negative"," bad"," poor"," awful"],[" positive"," good"," great"," awesome"," brilliant"," amazing"]],
    "mpqa": [[" terrible"," negative"," bad"," poor"," awful"],[" positive"," good"," great"," awesome"," brilliant"," amazing"]],
    "tweeteval_sentiment":[[" terrible"," negative"," bad"," poor"," awful"],[" moderate"," neutral"," balanced"],[" positive"," good"," great"," awesome"," brilliant"," amazing"]],
    "disaster":[[" no"," No"],[" yes"," Yes"]],
    "boolq":[[" A"," no"," No"],[" B"," yes", " Yes"]],
    "fpb": [[" terrible"," negative"," bad"," poor"," awful"],[" moderate"," neutral"," balanced"],[" positive"," good"," great"," awesome"," brilliant"," amazing"]],
    "imdb": [[" terrible"," negative"," bad"," poor"," awful"],[" positive"," good"," great"," awesome"," brilliant"," amazing"]],
    "mnli":[[" yes"," true"," Yes"," True"," correct"," Correct"," Right"," right"," Exactly"],[" maybe"," probably"," Probably"," Maybe"," possibly"," Perhaps"," perhaps"],[" no"," false"," False"," No"," wrong"," incorrect"]],
    "qnli":[[" yes"," true"," Yes"," True"," correct"," Correct"," Right"," right"," Exactly"],[" no"," false"," False"," No"," wrong"," incorrect"]],
    "snli":[[" yes"," true"," Yes"," True"," correct"," Correct"," Right"," right"," Exactly"],[" maybe"," probably"," Probably"," Maybe"," possibly"," Perhaps"," perhaps"],[" no"," false"," False"," No"," wrong"," incorrect"]],
    "agnews":[[" world"],[" sports"],[" business"],[" science"," technology"]],
    "arisetv":[[" business"],[" sports"],[" politics"],[" health"],[" entertainment"],[" technology"," science"]],
    "trec": [[" abbrevation"],[" entity"],[" description" " concept"], [" human"], [" location"],[" number"]],
    "dbpedia":[[" company"],[" school"," institute"],[" artist"],[" athlete"],[" officeholder"," civil servat"," official"],[" transportation"],[" building"],[" nature"],[" village"],[" animal"],[" plant"],[" album"],[" file"],[" writtenwork"," manuscript"]]
}

INIT_TEXT_DICT={
    "sst2": "Predict if sentiment of this review is positive or negative",
    "imdb":  "Predict if sentiment of this review is positive or negative",
    "mpqa": "Predict if sentiment of this text is positive or negative",
    "disaster": "Answer if this sentence is related to disaster or not",
    "tweeteval_sentiment": "Predict if sentiment of this tweet is positive, neutral or negative",
    "fpb": "Predict if sentiment of this financial news is positive, neutral or negative",
    "qnli": "Predict if the context sentence contains the answer to the question",
    "mnli": "Predict the inference relation between follwing two texts: entailment, contradiction, or neutral",
    "snli": "Predict the inference relation between follwing two texts: entailment, contradiction, or neutral",
    "agnews": "Predict the topic of the news",
    "arisetv": "Predict the topic of the news",
    "trec": "Predict the topic of the question",
    "boolq": "Choose the correct answer to the following question given the passage",
    "dbpedia": "Predict the ontology class of a given Wikipedia article"
}