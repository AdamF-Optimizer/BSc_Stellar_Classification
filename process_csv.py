import pandas as pd

def preprocess_stellar_data():
    """Clean and preprocess stellar classification data"""
    
    # Load the CSV file
    df = pd.read_csv("MSS_with_duplicates_afarrag.csv")
    
    # Drop duplicates and keep first occurrence
    df_unique = df.drop_duplicates(subset=['ra', 'dec'], keep='first').copy()
    
    # Read coordinates as floats
    df_unique['ra'] = df_unique['ra'].astype(float)
    df_unique['dec'] = df_unique['dec'].astype(float)
    
    # Filter to main sequence stars only
    main_classes = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
    filtered_data = df_unique[df_unique['subclass'].str.startswith(tuple(main_classes))].reset_index(drop=True)

    # Add main class column
    filtered_data['mainclass'] = filtered_data['subclass'].str[0]

    # Drop entries with null values for required fields
    filtered_data = filtered_data.dropna(subset=['rerun', 'run', 'camcol', 'field'])
    
    # Save cleaned data
    filtered_data.to_csv("MSS_with_duplicates_and_NULL_removed_afarrag.csv", index=False)
    
    return filtered_data

def prepare_camera_counts(df):
    """Prepare camera entry counts needed for dataset creation"""
    count_per_entry = df.groupby(['rerun', 'run', 'camcol', 'field', 'mainclass']).size().reset_index(name='count')
    count_per_entry_sorted = count_per_entry.sort_values(by='count', ascending=False)
    count_per_entry_sorted.to_csv('MSS_unique_cam_entry_counts_with_duplicates_and_NULL_removed_afarrag.csv', index=False)
    return count_per_entry_sorted

def create_datasets(count_df, clean_df, dataset_type='dataset_1'):
    if dataset_type == "dataset_1":
        grouped = count_df.groupby('mainclass')
        
        def select_entries(group, threshold):
            current_sum = 0
            sorted_group = group.sort_values('count', ascending=False).reset_index(drop=True)
            selected_rows = []
            
            for _, row in sorted_group.iterrows():
                if current_sum + row['count'] <= threshold:
                    selected_rows.append(row)
                    current_sum += row['count']
                elif current_sum < threshold:
                    # If adding the full count would exceed the threshold,
                    # we create a new row with the remaining count needed
                    # This results in sometimes taking a few entries over 10000
                    remaining = threshold - current_sum
                    new_row = row.copy()
                    new_row['count'] = remaining
                    selected_rows.append(new_row)
                    current_sum = threshold
                
                if current_sum == threshold:
                    break
            
            return pd.DataFrame(selected_rows)
        
        # Process each group
        result_list = []
        for name, group in grouped:
            if name == 'O':
                threshold = 3601
            else:
                threshold = 10000
            result_list.append(select_entries(group, threshold))
        
        # Combine all results to a single df
        result = pd.concat(result_list, ignore_index=True)
        
    elif dataset_type == "dataset_2" or dataset_type == "dataset_3":
        count_per_entry_sorted = count_df.sample(frac=1, random_state=42).reset_index(drop=True)
        target_count = 42000 if dataset_type == "dataset_2" else 75000 # ~42000 total stars for dataset 2, ~75000 for dataset 3
        sampled_entries = []
        current_count = 0
        
        for _, row in count_per_entry_sorted.iterrows():
            if current_count >= target_count:
                break
            entry_count = row['count']
            sampled_entries.append(row)
            current_count += entry_count
        
        result = pd.DataFrame(sampled_entries)

    # Save the camera data result
    result.to_csv(f"MSS_camera_data_{dataset_type}.csv", index=False)

    # Convert camera entries back to individual star entries
    unique_combinations = set(result[['rerun', 'run', 'camcol', 'field', 'mainclass']].itertuples(index=False, name=None))
    
    # Filter the larger DataFrame based on these combinations
    filtered_df = clean_df[clean_df.set_index(['rerun', 'run', 'camcol', 'field', 'mainclass']).index.isin(unique_combinations)]

    # Save the filtered result to a new CSV file
    filtered_df.to_csv(f"MSS_final_cleaned_camera_data_{dataset_type}.csv", index=False)
    
    return filtered_df

def split_train_val_test(df, dataset_type="dataset_1", train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1):
    """Split data into train/validation/test sets by class"""
    
    train_list = []
    validation_list = []
    test_list = []
    
    for mainclass in df['mainclass'].unique():
        class_df = df[df['mainclass'] == mainclass]
        class_df = class_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_end = int(train_ratio * len(class_df))
        validation_end = train_end + int(validation_ratio * len(class_df))
        
        train_list.append(class_df[:train_end])
        validation_list.append(class_df[train_end:validation_end])
        test_list.append(class_df[validation_end:])
    
    train_df = pd.concat(train_list).reset_index(drop=True)
    validation_df = pd.concat(validation_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    # Save splits
    train_df.to_csv(f"train_entries_{dataset_type}.csv", index=False)
    validation_df.to_csv(f"val_entries_{dataset_type}.csv", index=False)
    test_df.to_csv(f"test_entries_{dataset_type}.csv", index=False)
    
    return train_df, validation_df, test_df

clean_data = preprocess_stellar_data()

# Prepare camera counts for dataset creation
camera_counts = prepare_camera_counts(clean_data)

# Create all three datasets 
for dataset in ["dataset_1", "dataset_2", "dataset_3"]:
    print(f"Creating {dataset}...")
    final_df = create_datasets(camera_counts, clean_data, dataset)
    train_df, val_df, test_df = split_train_val_test(final_df, dataset)
    
    print(f"Dataset {dataset} completed:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples") 
    print(f"  Testing: {len(test_df)} samples")
    print(f"  Class distribution training: {train_df['mainclass'].value_counts().to_dict()}")
    print(f"  Class distribution validation: {val_df['mainclass'].value_counts().to_dict()}")
    print(f"  Class distribution testing: {test_df['mainclass'].value_counts().to_dict()}")
    print()
