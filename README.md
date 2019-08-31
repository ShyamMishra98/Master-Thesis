# Semantic Encoding for Evaluating Text Generation Models

### Instructions to run the source code

- All the necessary codes are in the folder source code.
- The test datasets are in the folder Testsets.
- The folder Human annotation consists of excel files of different datasets along with the human annotation scores.
- The codes are organized according to the semantic encoders.
- Each semantic encoder module inside the source code consists of codes to find the semantic similarity scores, lexical similarity scores and comparision of them with the human annotation scores 
- Each module has it's own instructions in the form of comments.
- The pretrained embeddings like GLOVE, Infersent can be downloaded using the links in the semantic encoders and should be kept in the folder 'Pretrained_embeddings'
- The links to download source codes from Github, which are necessary to run few semantic encoders are also in the beginning of those modules.
- The 'Testsets' folder consists of 5 different folders for the 5 different datasets used.
- Each dataset folder consists of the source testset, it's machine generated translations or summaries and the reference translations or summaries.
- The machine generated texts (Candidate texts) and reference text are used to find the similarity between them using semantic based encoders and lexical based measures.
- The semantic based encoders gives semantic similarity scores.
- Lexical based measures gives BLEU or ROUGE scores.
- The human annotation scores from the 'Human annotation' are read and compared with both the lexical and semantic based similarity scores.
- The Pearson correlation scores are used for comparision using the function 'pearsonr' from scipy.
- Higher the correlation score, higher is the similarity with the human annotations.



