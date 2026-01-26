def filter_valid_models(df_a, df_q, model_list=None, col_q_list=None,
                      col_model='model_id', col_task='task'):
    """
    In LiveBench, df_a represents either df_scores or df_answers of some set of models,
    and df_q represents a dataset containing various questions.

    Here, we will 
    - First merge some df_q columns into df_a
    - Filter in only models in model_list (or create model_list heuristically)
    """
    q_cols = df_q.columns
    if col_q_list is None:
        col_q_list = ['question_id', 'turns', 'c_i', 'livebench_release_date']
    if 'ground_truth' in q_cols:
        col_q_list += ['ground_truth']
    if 'subtask' in q_cols:
        col_q_list += ['subtask']
    df_merged = df_a.merge(
        df_q[col_q_list],
        on='question_id',
        how='inner'
        )

    if model_list is not None:
        df_merged = df_merged[df_merged[col_model].isin(model_list)]
        return df_merged

    # create model_list heuristically
    pivot_check = df_merged.pivot_table(index=col_model, columns='question_id', values=col_task, aggfunc='count')
    model_counts = pivot_check.sum(axis=1).sort_values()

    # Find the maximum number of questions answered by any model
    if not model_counts.empty:
        max_val = model_counts.max()
        filtered_models = model_counts[model_counts == max_val].index
    else:
        filtered_models = []

    df_merged = df_merged[df_merged[col_model].isin(filtered_models)]

    return df_merged

def perform_eda_checks(data_frame, model_str='model_id'):
    """ simple checks whether #rows == #questions x #models"""
    n_questions = data_frame['question_id'].nunique()
    n_models = data_frame[model_str].nunique()
    n_rows = len(data_frame)
    expected_rows = n_questions * n_models

    print(f"Total Unique Questions: {n_questions}")
    print(f"Total Unique Models: {n_models}")
    print(f"Actual Rows: {n_rows}")
    print(f"Expected Rows (if perfect): {expected_rows}")
    print(f"Missing Entries: {expected_rows - n_rows} ({((expected_rows-n_rows)/expected_rows)*100:.2f}%)")

def create_presence_table(data_frame, index_str='model_id', col_str='question_id', val_str='answer_id', aggfunc='count', force_int=False, debug=False):
    """
    data_frame is ideally "df_merged" of filter_valid_models, 
    where we will create an #models x # questions table 
    in which we can select value to represent existence ,  y_ij , b_ij , etc. [by specifying val_str and aggfunc]

    Example:
    Y_ij = create_presence_table(df_score_filtered, val_str='score', aggfunc='max')
    B_ij = create_presence_table(df_ans_filtered, val_str='B_ij', aggfunc='max')
    """
    df_presence_table = data_frame.pivot_table(
        index=index_str,
        columns=col_str,
        values=val_str,  # Use any column to count presence
        aggfunc=aggfunc       
    )

    # Fill NaN values (missing combinations) with -1
    df_presence_table = df_presence_table.fillna(-1)

    # Ensure all values are integers (1 or -1)
    if force_int:
        df_presence_table = df_presence_table.astype(int)
 
    # Verify counts
    if debug:
        num_present = (df_presence_table >= 1).sum().sum()
        num_missing = (df_presence_table == -1).sum().sum()

        print("Shape of the constructed presence table:", df_presence_table.shape)
        print(f"\nNumber of present entries (1): {num_present}")
        print(f"Number of missing entries (-1): {num_missing}")

    return df_presence_table

def reindex(Y, B, c, valid_model_names):
    """
    Ensure Y,B and c matrix have the same indices
    assuming model_id x question_id here
    """
    
    assert Y.shape == B.shape, "B and Y shapes must match"
    assert Y.shape[1] == len(c), "number of Y / c questions must match"

    assert(Y.notna().all().all()), "Y has some N/A"
    assert(B.notna().all().all()), "Y has some N/A"
    assert(c.notna().all().all()), "c has some N/A"
    
    assert list(Y.index) == valid_model_names, "Y's model_id sequence is wrong"
    assert list(B.index) == valid_model_names, "B's model_id sequence is wrong"
    
    if list(Y.columns) == list(B.columns):
        print("the question_id Y/B sequence is great!!")
    else:
        print("Swap B columns' order!!")
        B = B[Y.columns]
        
    if list(Y.columns) == list(c.index):
        print("the question_id Y/c sequence is great!!")
    else:
        print('reindex c!!')
        c = c.reindex(Y.columns)

    return Y, B, c

def filter_ans_and_score_data(df_ans_initial, df_score_initial, debug=True):
    """
    - Seclect indices in intersection of both df
    - create 
    """
    id_set1 = set(df_ans_initial.set_index(['model_id', 'question_id']).index)
    id_set2 = set(df_score_initial.set_index(['model_id', 'question_id']).index)

    if debug:
        print("*** Disable these messages by setting debug=False ***")
        print(f"Size of id_set1: {len(id_set1)}")
        print(f"Size of id_set2: {len(id_set2)}")
        print(f"Elements in id_set1 but not in id_set2: {len(id_set1-id_set2)}")
        print(f"Elements in id_set2 but not in id_set1: {len(id_set2-id_set1)}")

    id_intersection = id_set1.intersection(id_set2)
    if debug:
        print(f"Size of intersection: {len(id_intersection)}")

    df_ans_filtered = df_ans_initial[df_ans_initial.set_index(['model_id', 'question_id']).index.isin(id_intersection)]
    df_score_filtered = df_score_initial[df_score_initial.set_index(['model_id', 'question_id']).index.isin(id_intersection)]

    if debug:
        print(f"New shape of df_lang_merged after filtering: {df_ans_filtered.shape}")
        print(f"New shape of df_score_lang_merged after filtering: {df_score_filtered.shape}")

    # model_id x question_id for quick print investigation
    Y_ij = create_presence_table(df_score_filtered, val_str='score', aggfunc='max')
    B_ij = create_presence_table(df_ans_filtered, val_str='B_ij', aggfunc='max')

    # create c_i
    C_series = df_score_filtered.groupby('question_id')['c_i'].first()
    
    # Align C_series with Y_ij columns
    C_vector = C_series.reindex(Y_ij.columns).fillna(0.0)
    
    return df_ans_filtered, df_score_filtered, Y_ij, B_ij, C_vector

def calculate_c_i_rigorous(row):
    """
    Calculates Guessing Parameter (c_i) of each dataframe's row based on problem task / subtask.
    Valid for 5 categories: math, language, coding, instruction_following, reasoning
    only updated to public LiveBench on Huggingface (last updated on April, 2025)
    """

    task = row['task']
    
    if task == 'math_comp':
        if 'aime' in row['subtask']:
            # Integer 000-999 -> 1/1000
            return 0.001
        else:
            # AMC/SMC are 5-option Multiple Choice -> 1/5
            return 0.2
    if task == 'olympiad' and "ground_truth" in row.keys(): # IMO, USAMO
        from scipy.special import factorial
        n_elements = len(row["ground_truth"].split(','))

        # for trivial problems having n_elements==1 or 2, we decrease c_i a bit to reflect generative uncertainty
        # (it becomes resemble the classic reading comprehension task)
        if n_elements == 1: # avoid c_i = 1.0, which will hardly reflect real-world observations
            return 0.75
        if n_elements == 2:
            return 0.45

        return 1/factorial(n_elements) # approx. by gamma function
    
    if task == "web_of_lies_v2":
        return 0.125
    if task == "spatial":
        if "ground_truth" in row.keys() and row["ground_truth"] in ["square", "pentagon", "tetrahedron", "triangle", "tetrahedra", "square pyramid", "circle", "triangular prism", "sphere"]:
            return 1/7
        else:
            return 1/20
    if task == "zebra_puzzle":
        return 0.29 # ((1/3 + 1/4)/2) NOT a good way to measure performance
        
    
    if task == 'connections':
        # 1/70 or less
        return 0.01
    
    # AMPS_Hard (Latex generation) & Olympiad (Proof rearrangement)
    # Coding, IF, language (beside connections)
    # The probability of random guessing is negligible.    
    return 1e-5