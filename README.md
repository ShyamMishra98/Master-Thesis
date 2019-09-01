# Semantic Encoding for Evaluating Text Generation Models
The main goal of the thesis is to prove that the semantic based evaluation measures are better correlated with the human evaluation than the lexical based evaluation measures.

## Instructions

- All the necessary codes for automatic text generation are in the folder 'Automatic text generation models'.
- All the necessary codes for evaluation of automatically generated texts are in the folder 'Semantic encoders'.
- Each semantic encoder module consists of script to find the semantic similarity scores, lexical similarity scores and Pearson correlation scores.
- The test datasets are in the folder 'Testsets'.
- The folder 'Human annotations' consists of excel files of different datasets with the human annotation scores and a PDF file of annotation guidelines.

### Automatic text generation models
This folder contains models which were used to generate the texts automatically for machine translation and text summarization.

- The module 'OpenNMT DE-EN translation' contains source code for the translation from German to English using OpenNMT framework. 
- The module 'MarianNMT RO-EN translation' contains source code for the translation from Romanian to English using MarianNMT framework.
- The module 'OpenNMT Text Summarization' contains source code for the English text summarization for the CNN-DM dataset using OpenNMT framework.
- The module 'TensorFlow Text Summarization' contains source code for the English text summarization for the DUC2003, DUC2004 and Gigaword datasets using TensorFlow based model.

Each module has it's own instructions on how to run in the form of comments. It is advisable to run automatic text generation models in Google colab as it was original developed using it.

### Semantic encoders
The modules in semantic encoders are organized semantic encoders wise. Each module consists of the complete evaluation scripts of: semantic evaluation, lexical based evaluation and Pearson correlation evaluation.

- Each module has been developed in Jupyter notebooks and contains instructions to run in the form of comments.
- The links to download source codes from Github, which are necessary to run few semantic encoders are also in the beginning of those modules.
- The pretrained embeddings like GLOVE, Infersent can be downloaded using the links in the semantic encoders and must be kept in the folder 'Pretrained_embeddings'.
- The 'Testsets' folder consists of 5 different folders for the 5 different datasets used.
- Each dataset folder in the Testsets consists of the source testset, it's machine generated translations or summaries and the reference translations or summaries dataset.
- The similarity between the machine generated texts (Candidate texts) and reference text are computed, using semantic based encoders and lexical based measures.
- The semantic based encoders gives semantic similarity scores.
- Lexical based measures gives BLEU or ROUGE scores.
- The human annotation scores from the 'Human annotations' are read and compared with both the lexical and semantic based similarity scores.
- The Pearson correlation scores using the function 'pearsonr' from scipy is computed.
- Higher the correlation score, higher is the similarity with the human evaluation.

All the source code, testsets, human annotations can be found in the Github repository [Master-thesis](https://github.com/sanjita-suresh/Master-Thesis).



