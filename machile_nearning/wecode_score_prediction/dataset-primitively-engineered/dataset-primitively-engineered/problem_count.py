import pandas as pd
import logging
import os

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_assignment_stats(csv_file_path: str, output_filename: str = 'assignment_stats.csv') -> pd.DataFrame:
    """
    Loads submission data, counts unique problems and unique students
    per assignment ID, and saves the results to a CSV.

    Parameters:
    csv_file_path (str): Path to the input CSV file containing student submission data.
    output_filename (str): Name for the output CSV file.

    Returns:
    pd.DataFrame: DataFrame containing Assignment_ID, Unique_Problem_Count,
                  and Unique_Student_Count. Returns an empty DataFrame on failure.
    """
    logger.info(f"Loading data from {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
        logger.info(f"Successfully loaded {len(df)} records.")
    except FileNotFoundError:
        logger.error(f"Error: File not found at {csv_file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return pd.DataFrame()

    # Define the relevant column names based on your data's actual headers or patterns
    # Using the 'concat' patterns as they appear in your sample header
    assignment_col_raw = "concat('it001',`assignment_id`)"
    problem_col_raw = "concat('it001',`problem_id`)"
    username_col_raw = "concat('it001', username)" # Assuming this corresponds to student ID

    # Check if these columns exist in the dataframe
    if assignment_col_raw not in df.columns:
        logger.error(f"Error: Assignment ID column '{assignment_col_raw}' not found in the data.")
        return pd.DataFrame()
    if problem_col_raw not in df.columns:
        logger.error(f"Error: Problem ID column '{problem_col_raw}' not found in the data.")
        return pd.DataFrame()
    if username_col_raw not in df.columns:
         logger.error(f"Error: Username/Student ID column '{username_col_raw}' not found in the data.")
         # We can still proceed with problem count if username isn't strictly necessary
         # but for *student* count, it is. Let's make student count optional.
         username_col_exists = False
         logger.warning("Cannot calculate unique student count without the username column.")
    else:
         username_col_exists = True


    # Rename columns for easier use (cleaning the concat prefix is optional here,
    # as we just need consistent keys for grouping, but good for output)
    # Assuming the values themselves in the column are the actual IDs after loading.
    # If the values are still 'concat('it001',`id`)', you'd need a cleaning step here.
    # Let's assume the CSV reader handled the quoted/concat part and the column name is the 'concat' string.
    # If the column *names* are simple like 'assignment_id', update the raw names above.

    # Create mapping for cleaner output names
    clean_col_mapping = {
        assignment_col_raw: 'Assignment_ID',
        problem_col_raw: 'Problem_ID', # Keep problem col name for counting
        username_col_raw: 'Student_ID' # Keep student col name for counting
    }
    # Rename relevant columns in a copy to avoid modifying the original df if it's used elsewhere
    temp_df = df.rename(columns={k: v for k, v in clean_col_mapping.items() if k in df.columns})


    # --- 1. Count unique problem IDs per assignment ID ---
    logger.info("Counting unique problem IDs per assignment...")
    if 'Problem_ID' in temp_df.columns:
        problem_counts = temp_df.groupby('Assignment_ID')['Problem_ID'].nunique().reset_index()
        problem_counts.columns = ['Assignment_ID', 'Unique_Problem_Count']
        logger.info(f"Completed problem counts for {len(problem_counts)} assignments.")
    else:
         logger.warning("Skipping problem count as Problem_ID column not available after renaming.")
         problem_counts = pd.DataFrame(columns=['Assignment_ID', 'Unique_Problem_Count']) # Empty DataFrame


    # --- 2. Count unique students per assignment ID ---
    student_counts = pd.DataFrame(columns=['Assignment_ID', 'Unique_Student_Count']) # Initialize empty
    if username_col_exists and 'Student_ID' in temp_df.columns:
        logger.info("Counting unique student IDs per assignment...")
        # Need unique student-assignment pairs first, then group by assignment
        student_counts = temp_df[['Student_ID', 'Assignment_ID']].drop_duplicates().groupby('Assignment_ID').size().reset_index(name='Unique_Student_Count')
        student_counts.columns = ['Assignment_ID', 'Unique_Student_Count'] # Rename columns explicitly after size()
        logger.info(f"Completed student counts for {len(student_counts)} assignments.")
    else:
         logger.warning("Skipping student count as Student_ID column not available after renaming.")


    # --- 3. Merge the counts ---
    logger.info("Merging problem and student counts...")
    if not problem_counts.empty and not student_counts.empty:
        # Merge on Assignment_ID using an outer join to keep all assignments from both counts
        assignment_stats_df = pd.merge(
            problem_counts,
            student_counts,
            on='Assignment_ID',
            how='outer'
        )
    elif not problem_counts.empty:
        assignment_stats_df = problem_counts
        assignment_stats_df['Unique_Student_Count'] = 0 # Add student count column with 0 if only problem counts available
    elif not student_counts.empty:
        assignment_stats_df = student_counts
        assignment_stats_df['Unique_Problem_Count'] = 0 # Add problem count column with 0 if only student counts available
    else:
        logger.warning("No problem counts or student counts calculated. Resulting DataFrame is empty.")
        assignment_stats_df = pd.DataFrame(columns=['Assignment_ID', 'Unique_Problem_Count', 'Unique_Student_Count'])


    # Fill NaN counts with 0 after merging (for assignments present in one count but not the other)
    assignment_stats_df['Unique_Problem_Count'] = assignment_stats_df['Unique_Problem_Count'].fillna(0).astype(int)
    assignment_stats_df['Unique_Student_Count'] = assignment_stats_df['Unique_Student_Count'].fillna(0).astype(int)


    # --- 4. Output to CSV ---
    try:
        output_path = os.path.join(".", output_filename) # Save in current directory
        assignment_stats_df.to_csv(output_path, index=False)
        logger.info(f"Assignment statistics saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving assignment statistics to CSV: {e}")

    return assignment_stats_df

# --- Example Usage ---
if __name__ == "__main__":
    # Define the path to your raw submission log file
    # **IMPORTANT**: Update this path to your actual file location
    RAW_SUBMISSION_LOG_PATH = "annonimized.csv"

    # Define the output filename
    OUTPUT_STATS_FILENAME = "assignment_problem_student_counts.csv"

    # Run the analysis
    assignment_stats = analyze_assignment_stats(
        RAW_SUBMISSION_LOG_PATH,
        output_filename=OUTPUT_STATS_FILENAME
    )

    # Display top results or summary if needed
    if not assignment_stats.empty:
        logger.info("\n--- Assignment Statistics Summary ---")
        logger.info(f"Total unique assignments analyzed: {len(assignment_stats)}")

        # Sort by student count (most submitted to)
        top_by_students = assignment_stats.sort_values(by='Unique_Student_Count', ascending=False).head(10)
        logger.info("\nTop 10 Assignments by Unique Student Count:")
        print(top_by_students.to_string(index=False))

        # Sort by problem count (most problems in assignment)
        top_by_problems = assignment_stats.sort_values(by='Unique_Problem_Count', ascending=False).head(10)
        logger.info("\nTop 10 Assignments by Unique Problem Count:")
        print(top_by_problems.to_string(index=False))

        # Assignments with 1 student
        single_student_assignments = assignment_stats[assignment_stats['Unique_Student_Count'] == 1]
        logger.info(f"\nAssignments with exactly 1 unique student: {len(single_student_assignments)}")
        if not single_student_assignments.empty:
             # Print only the Assignment_ID and counts for brevity if there are many
             print(single_student_assignments.to_string(index=False))