from data_acquisition import *
from entity_resolution_pipeline_blocking import *
from entity_resolution_pipeline_matching import *
from data_preparation import *

if __name__ == '__main__':
    """
    # --Data acquisition--
    df_dblp = read_data("data/dblp.txt", encoding='utf-8"')
    df_dblp = filter_data(df_dblp)
    df_dblp.to_csv('Data_filtered/DBLP 1995 2004.csv', index=False)
    df_acm = read_data("data/citation-acm-v8.txt", encoding='utf-8"')

    # --Data preparation--
    df_acm = filter_data(df_acm)
    df_acm.to_csv('Data_filtered/ACM 1995 2004.csv', index=False)
    """

    # READ FILES
    df_acm = pd.read_csv('Data_filtered/ACM 1995 2004.csv').head(500)
    df_dblp = pd.read_csv('Data_filtered/DBLP 1995 2004.csv').head(500)

    # --Blocking structured keys with "Year column"--
    key_blocks_structured_keys = divide_blocks_structured_keys(df1=df_acm,
                                                               df2=df_dblp,
                                                               column_name="Year")
    """
    # --Blocking ngrams with n=10 and column="Authors"--
    key_blocks_n_grams = divide_blocks_n_gram_blocking(df1=df_acm, df2=df_dblp, column="Authors", n=10)
    """

    # --Matching Preparation--
    # Prepare text data for fitting the vectorizer
    text_data_acm = df_acm['Title'].fillna('') + " " + df_acm['Authors'].fillna('')
    text_data_dblp = df_dblp['Title'].fillna('') + " " + df_dblp['Authors'].fillna('')
    combined_text_data = pd.concat([text_data_acm, text_data_dblp]).unique()

    # Fit the global vectorizer
    global_vectorizer.fit(combined_text_data)

    # create a baseline pipeline
    similar_pairs_baseline = baseline_pipeline(df_acm, df_dblp,
                                               similarity_threshold=0.9,
                                               similarity_metric='Jaccard')
    """ 
    write_similar_pairs_to_csv(similar_pairs=similar_pairs_baseline, file_name="baseline.csv")
    
    # --ngrams matching--
     similar_pairs_tf_idf_n_grams = row_matching_ngrams(blocks=key_blocks_n_grams,
                                                 similarity_threshold=0.9,
                                                 similarity_metric='TF-IDF')
     similar_pairs_jacquard_n_grams = row_matching_ngrams(blocks=key_blocks_n_grams,
                                                   similarity_threshold=0.9,
                                                   similarity_metric='Jaccard')
     similar_pairs_levenshtein_n_grams = row_matching_ngrams(key_blocks_n_grams,
                                                      similarity_threshold=0.9,
                                                      similarity_metric='Levenshtein')
    """

    # --structured keys matching--
    similar_pairs_jacquard_structured_keys = row_matching_structured_keys(blocks=key_blocks_structured_keys,
                                                                          column_name="Year",
                                                                          similarity_threshold=0.9,
                                                                          similarity_metric='Jaccard')

    precision, recall, f1_score = calculate_metrics(base_pairs=similar_pairs_baseline,
                                                    comparison_pairs=similar_pairs_jacquard_structured_keys)

    print(precision, recall, f1_score)
    """ 
    similar_pairs_jacquard_structured_keys = row_matching_structured_keys(blocks=key_blocks_structured_keys,
                                                                          column_name="Year",
                                                                          similarity_threshold=0.9,
                                                                          similarity_metric='Jaccard')
    similar_pairs_levenshtein_structured_keys = row_matching_structured_keys(blocks=key_blocks_structured_keys,
                                                                             column_name="Year",
                                                                             similarity_threshold=0.9,
                                                                             similarity_metric='Levenshtein')
    similar_pairs_tf_idf_structured_keys = row_matching_structured_keys(blocks=key_blocks_structured_keys,
                                                                        similarity_threshold=0.9,
                                                                        column_name="Year",
                                                                        similarity_metric='TF-IDF')
    """
