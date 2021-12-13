library(foreign)
library(psych)
library(NLP) 
library(tm)
library(ldatuning)
library(slam)
library(lsa) 

#example
jianghu_imdb <- read.csv(file.choose(), header = TRUE, encoding = "UTF-8", stringsAsFactors=FALSE)

#data cleaning 
jianghu_imdb_docs <- subset(jianghu_imdb, select = c("reviews_number", "reviews_content")) 
jianghu_imdb_VCorpus <- VCorpus(VectorSource(jianghu_imdb_docs)) 

removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
jianghu_imdb_VCorpus  <- tm_map(jianghu_imdb_VCorpus, content_transformer(removeURL)) 
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
jianghu_imdb_VCorpus <- tm_map(jianghu_imdb_VCorpus, content_transformer(removeNumPunct)) 
stopwords("english")  
jianghu_imdb_VCorpus <- tm_map(jianghu_imdb_VCorpus, removeWords, stopwords("english"))
jianghu_imdb_VCorpus <- tm_map(jianghu_imdb_VCorpus, stripWhitespace) 
jianghu_imdb_VCorpus <- tm_map(jianghu_imdb_VCorpus, removePunctuation)
library(SnowballC)
jianghu_imdb_VCorpus <- tm_map(jianghu_imdb_VCorpus, stemDocument)  

#TF-IDF
jianghu_imdb_dtm <- DocumentTermMatrix(jianghu_imdb_VCorpus, control = list(removePunctuation = TRUE, stopwords=TRUE)) 
jianghu_imdb_dtm
inspect(jianghu_imdb_dtm)
term_freq_jianghu_imdb <- colSums(as.matrix(jianghu_imdb_dtm))  
write.csv(as.data.frame(sort(rowSums(as.matrix(term_freq_jianghu_imdb)), decreasing=TRUE)), file="jianghu_imdb_TF.csv")

jianghu_imdb_dtm_tfidf <- DocumentTermMatrix(jianghu_imdb_VCorpus, control = list(weighting = weightTfIdf)) 
print(jianghu_imdb_dtm_tfidf) 
jianghu_imdb_dtm_tfidf2 = removeSparseTerms(jianghu_imdb_dtm_tfidf, 0.99) 
print(jianghu_imdb_dtm_tfidf2) 
write.csv(as.data.frame(sort(colSums(as.matrix(jianghu_imdb_dtm_tfidf2)), decreasing=TRUE)), file="jianghu_imdb_TFIDF.csv")

library(topicmodels) 

# clean the empty (non-zero entry) 
jianghu_imdb <- apply(jianghu_imdb_dtm , 1, sum) 
jianghu_imdb_dtm_nonzero <- jianghu_imdb_dtm[jianghu_imdb> 0, ]

# LDA topic model k10 - 10 topics, 10 term   
jianghu_imdb_dtm_10topics <- LDA(jianghu_imdb_dtm_nonzero, k = 10, method = "Gibbs", control = list(iter=2000, seed = 2000)) 
jianghu_imdb_dtm_10topics_10words <- terms(jianghu_imdb_dtm_10topics, 20) 
(jianghu_imdb_dtm_10topics_10words <- apply(jianghu_imdb_dtm_10topics_10words, MARGIN = 2, paste, collapse = ", "))


