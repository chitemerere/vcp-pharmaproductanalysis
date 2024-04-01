#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from datetime import datetime 
import os
import toml
import chardet

# Specify the timezone for Harare
harare_timezone = ZoneInfo("Africa/Harare")

# Initialize session state variables at the start
def initialize_session_state():
    if 'start_time' not in st.session_state:
        st.session_state['start_time'] = None
    if 'end_time' not in st.session_state:
        st.session_state['end_time'] = None
    if 'processed_rows' not in st.session_state:
        st.session_state['processed_rows'] = 0
    if 'fuzzy_matched_data' not in st.session_state:
        st.session_state['fuzzy_matched_data'] = pd.DataFrame()

initialize_session_state()

# Define a key for your upload in session_state to check if the data is already loaded
data_key = 'ema_fda_healthcanada_data'

def apply_all_filters(df, filter_settings):
    """
    Apply all filters to the dataframe based on the filter settings.
    """
    # Year range filter
    if 'year_range' in filter_settings:
        start_year, end_year = filter_settings['year_range']
        df = df[(df['Approval Year'] >= start_year) & (df['Approval Year'] <= end_year)]

    # NDA/BLA filter
    if filter_settings['nda_bla_selection'] != 'All':
        df = df[df['NDA/BLA'] == filter_settings['nda_bla_selection']]

    # Active Ingredient/Moiety filter
    if filter_settings['active_ingredient_selection'] != 'All':
        df = df[df['Active Ingredient/Moiety'] == filter_settings['active_ingredient_selection']]

    # Review Designation filter
    if filter_settings['review_designation_selection'] != 'All':
        df = df[df['Review Designation'] == filter_settings['review_designation_selection']]

    # Boolean filters (Yes/No or presence checks)
    if filter_settings['orphan_drug_option']:
        df = df[df['Orphan Drug Designation'] == 'Yes']
    if filter_settings['accelerated_approval_option']:
        df = df[df['Accelerated Approval'] == 'Yes']
    if filter_settings['breakthrough_therapy_option']:
        df = df[df['Breakthrough Therapy Designation'] == 'Yes']
    if filter_settings['fast_track_option']:
        df = df[df['Fast Track Designation'] == 'Yes']
    if filter_settings['qualified_infectious_option']:
        df = df[df['Qualified Infectious Disease Product'] == 'Yes']

    return df

def safe_load_csv(uploaded_file):
    if uploaded_file is not None and uploaded_file.size > 0:
        return pd.read_csv(uploaded_file)
    else:
        return None

# Function to load data
@st.cache_data

def load_data(uploaded_file):
    if uploaded_file is not None:
        # Check if the uploaded file is not empty
        uploaded_file.seek(0)  # Go to the start of the file
        if uploaded_file.read(1024) == b'':  # Read the first 1KB to check if it's empty
            st.error('The uploaded file is empty.')
            return pd.DataFrame()
        
        # Reset the file pointer to the start of the file after checking
        uploaded_file.seek(0)
        
        # Since the file is not empty, attempt to read it as a CSV
        df = pd.read_csv(uploaded_file)
        
        # After loading, check if the DataFrame is actually empty or if there's a 'Generic Name' column
        if df.empty:
            st.error('The uploaded file is empty or not in a valid CSV format.')
            return pd.DataFrame()
        
        if 'Generic Name' in df.columns:
            df['Generic Name'] = df['Generic Name'].str.upper().str.strip()
        else:
            st.error("'Generic Name' column not found in the uploaded file.")
            return pd.DataFrame()

        return df
    else:
        # If no file was uploaded, return an empty DataFrame
        return pd.DataFrame()
    
# Adjusted ATC code extraction functions
def extract_atc_levels_human(atc_code):
    # Ensure atc_code is a string to prevent TypeError
    atc_code = str(atc_code) if pd.notna(atc_code) else ""
    return pd.Series([atc_code[:1], atc_code[:3], atc_code[:4], atc_code[:5]])

def extract_atc_levels_veterinary(atc_code):
    atc_code = str(atc_code) if pd.notna(atc_code) else ""
    return pd.Series([atc_code[:1], atc_code[:3], atc_code[:4], atc_code[:5]])
        
# Function to convert DataFrame to CSV
def convert_df_to_csv(df):
    if df is not None:
        return df.to_csv(index=False).encode('utf-8')
    else:
        # Handle the case where df is None, e.g., return an empty string or None
        return None

# Calculation Functions with Corrected Factors and Rounding
def calculate_prevalent_population(population, prevalence):
    return round(population * prevalence / 100, 2)

def calculate_symptomatic_population(prevalent_population, symptomatic_rate):
    return round(prevalent_population * symptomatic_rate / 100, 2)

def calculate_diagnosed_population(symptomatic_population, diagnosis_rate):
    return round(symptomatic_population * diagnosis_rate / 100, 2)

def calculate_potential_patients(diagnosed_population, access_rate):
    return round(diagnosed_population * access_rate / 100, 2)

def calculate_drug_treated_patients(potential_patients, treatment_rate):
    return round(potential_patients * treatment_rate / 100, 2)

def load_and_process_prohibited_generics(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Ensure 'Generic Name' is in uppercase to match the mcaz_register format
        df['Generic Name'] = df['Generic Name'].str.upper()
        return df
    return pd.DataFrame()

def filter_data_for_user(user_type, merged_data, prohibited_list):
    if user_type == 'Importer':
        # Create a temporary column to identify rows to be filtered out
        # Merge merged_data with prohibited_list on both 'Generic Name' and 'Form'
        # Use an indicator to identify rows that exist in both DataFrames
        temp_merged = merged_data.merge(prohibited_list, on=['Generic Name', 'Form'], how='left', indicator=True)
        
        # Filter out rows that are found in the prohibited_list (i.e., those with '_merge' == 'both')
        filtered_data = temp_merged[temp_merged['_merge'] == 'left_only']
        
        # Drop the '_merge' column as it's no longer needed
        filtered_data = filtered_data.drop(columns=['_merge'])
    else:
        # Local Manufacturer gets the full data
        filtered_data = merged_data
    
    return filtered_data

def apply_mutually_exclusive_filters(data, filters):
    for key, selected in filters.items():
        if selected and selected != 'None':
            data = data[data[key] == selected]
    return data

# Function for fuzzy matching of principal names
def fuzzy_match_names(series, threshold=90):
    # Convert the series to string type and fill NaN values with an empty string
    series = series.fillna('').astype(str)
    unique_names = series.unique()
    matched_names = {}
    
    for name in unique_names:
        if name:  # Check if the name is not an empty string
            # Find the best match for each unique name
            best_match = process.extractOne(name, unique_names, scorer=fuzz.token_sort_ratio)
            if best_match[1] >= threshold:
                matched_names[name] = best_match[0]
            else:
                matched_names[name] = name
        else:
            matched_names[name] = name  # Keep empty strings as is

    return series.map(matched_names)

def load_data_fda(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return pd.DataFrame() 

def filter_fda_data(fda_data, mcaz_register):
    filtered_data = fda_data.copy()
    for index, row in fda_data.iterrows():
        if ((mcaz_register['Generic Name'] == row['ACTIVE INGREDIENT']) &
            (mcaz_register['Strength'] == row['DOSAGE STRENGTH']) &
            (mcaz_register['Form'] == row['DOSAGE FORM'])).any():
            filtered_data = filtered_data.drop(index)
    return filtered_data

def load_data_orange(file):
    if file is not None:
        return pd.read_csv(file)
    return pd.DataFrame()

def outer_join_dfs(df1, df2, df3, key):
    return df1.merge(df2, on=key, how='outer').merge(df3, on=key, how='outer')

def filter_dataframe(df, column, value):
    if value != "None":
        return df[df[column] == value]
    return df

# Function to load data from an uploaded file
@st.cache_data
def load_data_sales(uploaded_file):
    if uploaded_file is not None:
        try:
            # Attempt to read the uploaded file with specified encoding
            df = pd.read_csv(uploaded_file, encoding='latin1')
            # Check if the file is empty by looking at the DataFrame shape
            if df.empty:
                st.error("Uploaded file is empty. Please upload a file with data.")
                return pd.DataFrame()  # Return an empty DataFrame as a fallback
            return df
        except pd.errors.EmptyDataError:
            # Handle the case where the CSV file is empty or not properly formatted
            st.error("Uploaded file is empty or not properly formatted. Please check the file and try again.")
            return pd.DataFrame()  # Return an empty DataFrame as a fallback
        except UnicodeDecodeError as e:
            # Handle potential Unicode decoding errors by providing a message to the user
            st.error(f"Error decoding file: {e}. Try changing the file encoding and upload again.")
            return pd.DataFrame()
        except Exception as e:
            # Handle any other exceptions that may occur
            st.error(f"An error occurred while processing the file: {e}")
            return pd.DataFrame()
    else:
        # If no file is uploaded, return an empty DataFrame
        return pd.DataFrame()
    
# Function to load a CSV file into a DataFrame with caching
@st.cache_data
def load_file(file):
    return pd.read_csv(file)

# Function to initialize necessary columns in the DataFrame if they don't exist
def init_columns(df):
    for column in ['Best Match Name', 'Match Score', 'ATCCode']:
        if column not in df.columns:
            df[column] = pd.NA
    return df

def process_data(mcaz_register, atc_index, extract_atc_levels):
     # Ensure start_time is set at the beginning of processing
    if 'start_time' not in st.session_state or st.session_state.start_time is None:
        st.session_state.start_time = datetime.now(harare_timezone)  # or datetime.now(harare_timezone) if timezone is relevant
        
    # Ensure 'Name' in ATC index is string
    atc_index['Name'] = atc_index['Name'].astype(str)

    name_to_atc_code = dict(zip(atc_index['Name'], atc_index['ATCCode']))
    total_rows = len(mcaz_register)
    processed_rows = st.session_state.get('processed_rows', 0)

    total_rows = len(mcaz_register)
    processed_rows = st.session_state.get('processed_rows', 0)

    # Initialize progress bar and display processing message
    progress_bar = st.progress(0)
    st.subheader('Processing and mapping data...')
    st.session_state.start_time = datetime.now(harare_timezone) 
    st.write(f"Processing started at: {st.session_state.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for index, row in mcaz_register.iloc[processed_rows:].iterrows():
        # Processing logic (omitted for brevity)
        if not st.session_state.get('resume_processing', True):
            break  # Pause processing
        
        match_result = process.extractOne(row['Generic Name'], atc_index['Name'], scorer=fuzz.ratio)
        best_match_name, match_score = match_result[0], match_result[1] if match_result else (None, 0)
        atc_code = name_to_atc_code.get(best_match_name, None)

        mcaz_register.at[index, 'Best Match Name'] = best_match_name
        mcaz_register.at[index, 'Match Score'] = match_score
        mcaz_register.at[index, 'ATCCode'] = atc_code

        progress = int(((index - processed_rows + 1) / (total_rows - processed_rows)) * 100)
        progress_bar.progress(progress)
        st.session_state.processed_rows = index + 1
        
        # Update progress
        progress = int(((index - processed_rows + 1) / total_rows) * 100)
        progress_bar.progress(progress)
        st.session_state.processed_rows = index + 1

    # Finalize progress and display completion message
    progress_bar.progress(100)
    st.session_state.end_time = datetime.now(harare_timezone)
    # Safely calculate processing time
    if st.session_state.start_time is not None and st.session_state.end_time is not None:
        processing_time = st.session_state.end_time - st.session_state.start_time
        st.write(f"Processing completed at: {st.session_state.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"Total processing time: {processing_time}")
    else:
        st.error("Processing time could not be calculated due to missing start or end time.")
    st.session_state.fuzzy_matched_data = mcaz_register  # Save processed data for later use
    
def process_data_fda(fda_register, atc_index, extract_atc_levels):
     # Ensure start_time is set at the beginning of processing
    if 'start_time' not in st.session_state or st.session_state.start_time is None:
        st.session_state.start_time = datetime.now(harare_timezone)  # or datetime.now(harare_timezone) if timezone is relevant
        
    # Ensure 'Name' in ATC index is string
    atc_index['Name'] = atc_index['Name'].astype(str)

    name_to_atc_code = dict(zip(atc_index['Name'], atc_index['ATCCode']))
    total_rows = len(fda_register)
    processed_rows = st.session_state.get('processed_rows', 0)

    # Initialize progress bar and display processing message
    progress_bar = st.progress(0)
    st.subheader('Processing and mapping data...')
    st.session_state.start_time = datetime.now(harare_timezone) 
    st.write(f"Processing started at: {st.session_state.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

#     for index, row in fda_register.iloc[processed_rows:].iterrows():
    for index, row in st.session_state.fda_register.iterrows(): 
        # Processing logic (omitted for brevity)
        if not st.session_state.get('resume_processing', True):
            break  # Pause processing
        
        match_result = process.extractOne(row['Ingredient'], atc_index['Name'], scorer=fuzz.ratio)
        best_match_name, match_score = match_result[0], match_result[1] if match_result else (None, 0)
        atc_code = name_to_atc_code.get(best_match_name, None)
        
        fda_register.at[index, 'Best Match Name'] = best_match_name
        fda_register.at[index, 'Match Score'] = match_score
        fda_register.at[index, 'ATCCode'] = atc_code
        
        progress = int(((index - processed_rows + 1) / (total_rows - processed_rows)) * 100)
        progress_bar.progress(progress)
        st.session_state.processed_rows = index + 1
        
        # Update progress
        progress = int(((index - processed_rows + 1) / total_rows) * 100)
        progress_bar.progress(progress)
        st.session_state.processed_rows = index + 1
  
    # Finalize progress and display completion message
    progress_bar.progress(100)
    st.session_state.end_time = datetime.now(harare_timezone)
    # Safely calculate processing time
    if st.session_state.start_time is not None and st.session_state.end_time is not None:
        processing_time = st.session_state.end_time - st.session_state.start_time
        st.write(f"Processing completed at: {st.session_state.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"Total processing time: {processing_time}")
    else:
        st.error("Processing time could not be calculated due to missing start or end time.")
    st.session_state.fuzzy_matched_data = fda_register  # Save processed data for later use
    
def check_required_columns(df, required_columns, level):
    """
    Checks if the required columns are present in the dataframe.
    If not, displays a warning message and updates the session state to indicate the check failed.
    """
    if df is not None:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.warning(f"Missing required columns for ATC Level {level}: {', '.join(missing_columns)}")
            st.session_state['check_passed'] = False
        if not missing_columns:
            st.success(f"All required column for ATC Level {level} are present.")
    else:
        st.warning(f"No file uploaded for ATC Level {level}.")
        st.session_state['check_passed'] = False
    
# Function to check for required columns in the uploaded file
def check_required_columns_in_file(file, required_columns):
    if file is not None:
        # Attempt to read the uploaded file into a DataFrame
        try:
            df = pd.read_csv(file)
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                return False, missing_columns
            return True, None
        except Exception as e:
            st.error(f"Failed to read the uploaded file. Error: {str(e)}")
            return False, None
    return None, None  # Indicates no file was uploaded

def check_prohibited_file_columns(df, required_columns):
    # Check for the presence of required columns in the dataframe
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        return False, missing_columns
    return True, []

# Helper function to process the uploaded file and generate the "COUNTRY" column
def process_uploaded_file(uploaded_file):
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    df['COUNTRY_CODE'] = df['ADDRESS'].str.extract(r'\(([^)]+)\)$')
    columns = [
        "FIRM_NAME",
        "ADDRESS",
        "COUNTRY_CODE",
        "EXPIRATION_DATE",
        "OPERATIONS",
        "ESTABLISHMENT_CONTACT_NAME",
        "ESTABLISHMENT_CONTACT_EMAIL",
        "REGISTRANT_NAME",
        "REGISTRANT_CONTACT_NAME",
        "REGISTRANT_CONTACT_EMAIL"
    ]
    df = df[columns]
    return df

# Define the filter_dataframe function with the 'Country' filter instead of 'Country Code'
def filter_dataframe_establishments(df, firm_name, country, operations, registrant_name):
    if firm_name != "All":
        df = df[df['FIRM_NAME'] == firm_name]
    if country != "All":
        df = df[df['Country'] == country]  # Changed to 'Country'
    if operations != "All":
        df = df[df['OPERATIONS'].apply(lambda x: x.strip() == operations)]

    if registrant_name != "All":
        df = df[df['REGISTRANT_NAME'] == registrant_name]
    
    # Sort the dataframe
    return df.sort_values(by=["FIRM_NAME", "Country", "OPERATIONS", "REGISTRANT_NAME"], ascending=True)

def load_data_nme(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin1')

        # Explicitly check if 'FDA Approval Date' is already in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['FDA Approval Date']):
            try:
                df['FDA Approval Date'] = pd.to_datetime(df['FDA Approval Date'], errors='coerce')
            except Exception as e:
                st.error(f"Error converting FDA Approval Date to datetime: {e}")
                return None

        # Ensure the conversion was successful by checking for datetime dtype again
        if pd.api.types.is_datetime64_any_dtype(df['FDA Approval Date']):
            df['Approval Year'] = df['FDA Approval Date'].dt.year.dropna().astype(int)
        else:
            st.error("Failed to convert 'FDA Approval Date' to datetime format.")
            return None

        return df
    else:
        return None
    
# This function now returns an HTML <a> tag for each link
def construct_espacenet_link(patent_no):
    espacenet_base_url = "https://worldwide.espacenet.com/searchResults?submitted=true&locale=en_EP&DB=EPODOC&ST=advanced&TI=&AB=&PN="
    link = f"{espacenet_base_url}{patent_no}&AP=&PR=&PD=&PA=&IN=&CPC=&IC=&Submit=Search"
    return f'<a href="{link}" target="_blank">{patent_no}</a>'

def construct_wipo_link(patent_no):
    # This is a base URL for initiating a search on WIPO. Adjustments might be needed based on the exact requirement.
    wipo_search_base_url = "https://patentscope.wipo.int/search/en/search.jsf"
    # The query parameter 'searchQuery' is assumed to be the way to pre-fill the search; adjust based on actual parameter names.
    # Note: This is speculative and may not work as expected without the correct parameter names and values.
    link = f"{wipo_search_base_url}?searchQuery={patent_no}"
    # Return the HTML anchor tag for the link
    return f'<a href="{link}" target="_blank">{patent_no}</a>'

# Function to check if required columns are present in the dataframe
def check_required_columns_orangebook(df, required_columns):
    if df is None:
        return False, ["DataFrame is None"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        return False, missing_columns
    return True, None

def check_columns(uploaded_file, required_columns):
    """Check if uploaded file contains all required columns."""
    try:
        # Read a small part of the file to determine its encoding
        rawdata = uploaded_file.read(10000)  # Read the first 10,000 bytes to detect encoding
        uploaded_file.seek(0)  # Reset file pointer to the beginning
        result = chardet.detect(rawdata)
        
        encoding = result['encoding']
        if encoding == 'ascii' or not encoding:
            encoding = 'Windows-1252'  # Fallback to 'Windows-1252' if 'ascii' or detection failed
        
        # Attempt to read the file with the detected or fallback encoding
        try:
            df = pd.read_csv(uploaded_file, encoding=encoding)
        except UnicodeDecodeError:
            uploaded_file.seek(0)  # Reset file pointer and try with 'Windows-1252'
            df = pd.read_csv(uploaded_file, encoding='Windows-1252')
        
        if all(column in df.columns for column in required_columns):
            return df
        else:
            st.error(f"{uploaded_file.name} does not contain all required columns.")
            return None
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {str(e)}")
        return None

def process_data_Drugs(dataframes):
    # Join operations
    products_df = dataframes["Products@FDA.csv"]
    applications_df = dataframes["Applications.csv"]
    submissions_df = dataframes["Submissions.csv"]
    marketing_status_df = dataframes["MarketingStatus.csv"]
    marketing_status_lookup_df = dataframes["MarketingStatus_Lookup.csv"]

    # Join Products on Applications, Submissions, and MarketingStatus
    merged_df = products_df.merge(applications_df, on="ApplNo", how="left") \
                            .merge(submissions_df, on="ApplNo", how="left") \
                            .merge(marketing_status_df, on=["ApplNo", "ProductNo"], how="left") \
                            .merge(marketing_status_lookup_df, on="MarketingStatusID", how="left")

    # Drop values where MarketingStatusID is 3 or 5
    merged_df = merged_df[~merged_df.MarketingStatusID.isin([3, 5])]
    
    columns_to_drop = ['ApplPublicNotes', 'SubmissionClassCodeID', 'SubmissionNo', 'SubmissionsPublicNotes', 'MarketingStatusID']
    columns_to_drop = [col for col in columns_to_drop if col in merged_df.columns]

    # Drop the columns from the dataframe safely
    merged_df.drop(columns=columns_to_drop, axis=1, inplace=True)

    return merged_df

def perform_drugs_fda_analysis():
    st.subheader('Drugs@FDA Analysis')

    # Check if the data has already been processed and stored in session state
    if 'processed_data_Drugs' not in st.session_state:
        uploaded_files = st.file_uploader(
            "Upload files",
            accept_multiple_files=True,
            help="Upload the files: Products@FDA.csv, Applications.csv, Submissions.csv, MarketingStatus.csv, MarketingStatus_Lookup.csv"
        )

        if uploaded_files:
            files = {file.name: file for file in uploaded_files}
            required_columns = {
                # Your required columns here
                "Products@FDA.csv": ["ApplNo", "ProductNo", "Form", "Strength", "ReferenceDrug", "DrugName", "ActiveIngredient", "ReferenceStandard"],
                "Applications.csv": ["ApplNo", "ApplType", "ApplPublicNotes", "SponsorName"],
                "Submissions.csv": ["ApplNo", "SubmissionClassCodeID", "SubmissionType", "SubmissionNo", "SubmissionStatus", "SubmissionStatusDate", "SubmissionsPublicNotes", "ReviewPriority"],
                "MarketingStatus.csv": ["ApplNo", "ProductNo", "MarketingStatusID"],
                "MarketingStatus_Lookup.csv": ["MarketingStatusID", "MarketingStatusDescription"],            
            }

            dataframes = {}
            for filename, required_cols in required_columns.items():
                if filename in files:
                    df = check_columns(files[filename], required_cols)
                    if df is not None:
                        dataframes[filename] = df
                else:
                    st.warning(f"{filename} not uploaded.")

            if len(dataframes) == 5:
                # Process data and store it in session state
                st.session_state['processed_data_Drugs'] = process_data_Drugs(dataframes)

    if 'processed_data_Drugs' in st.session_state:
        merged_df = st.session_state['processed_data_Drugs']
        # Continue with displaying the processed data or further analysis...
        # Mutually Exclusive Filters
        form_options = ['All'] + sorted(merged_df['Form'].unique().tolist())
        drug_name_options = ['All'] + sorted(merged_df['DrugName'].unique().tolist())
        active_ingredient_options = ['All'] + sorted(merged_df['ActiveIngredient'].unique().tolist())
        appl_type_options = ['All'] + sorted(merged_df['ApplType'].astype(str).unique().tolist())
        sponsor_name_options = ['All'] + sorted(merged_df['SponsorName'].astype(str).unique().tolist())
        marketing_status_options = ['All'] + sorted(merged_df['MarketingStatusDescription'].astype(str).unique().tolist())
        submission_type_options = ['All'] + sorted(merged_df['SubmissionType'].astype(str).unique().tolist())
        review_priority_options = ['All'] + sorted(merged_df['ReviewPriority'].astype(str).unique().tolist())
        submission_status_options = ['All'] + sorted(merged_df['SubmissionStatus'].astype(str).unique().tolist())

        form_selection = st.selectbox("Form", options=form_options)
        drug_name_selection = st.selectbox("Drug Name", options=drug_name_options)
        active_ingredient_selection = st.selectbox("Active Ingredient", options=active_ingredient_options)
        appl_type_selection = st.selectbox("Application Type", options=appl_type_options)
        sponsor_name_selection = st.selectbox("Sponsor Name", options=sponsor_name_options)
        marketing_status_selection = st.selectbox("Marketing Status", options=marketing_status_options)
        submission_type_selection = st.selectbox("Submission Type", options=submission_type_options)
        review_priority_selection = st.selectbox("Review Priority", options=review_priority_options)
        submission_status_selection = st.selectbox("Submission Status", options=submission_status_options)

        # Apply filters
        if form_selection != 'All':
            merged_df = merged_df[merged_df['Form'] == form_selection]
        if drug_name_selection != 'All':
            merged_df = merged_df[merged_df['DrugName'] == drug_name_selection]
        if active_ingredient_selection != 'All':
            merged_df = merged_df[merged_df['ActiveIngredient'] == active_ingredient_selection]
        if appl_type_selection != 'All':
            merged_df = merged_df[merged_df['ApplType'] == appl_type_selection]
        if sponsor_name_selection != 'All':
            merged_df = merged_df[merged_df['SponsorName'] == sponsor_name_selection]
        if marketing_status_selection != 'All':
            merged_df = merged_df[merged_df['MarketingStatusDescription'] == marketing_status_selection]
        if submission_type_selection != 'All':
            merged_df = merged_df[merged_df['SubmissionType'] == submission_type_selection]
        if review_priority_selection != 'All':
            merged_df = merged_df[merged_df['ReviewPriority'] == review_priority_selection]
        if submission_status_selection != 'All':
            merged_df = merged_df[merged_df['SubmissionStatus'] == submission_status_selection]
        
        st.dataframe(merged_df)
        st.write(f"Filtered data count: {len(merged_df)}")
        # Example: Download button for processed data
        csv = convert_df_to_csv(merged_df)
        st.download_button(
            label="Download processed data as CSV",
            data=csv,
            file_name='processed_data_drugs@fda.csv',
            mime='text/csv',
        )


def display_main_application_content():
                        
    # Initialize mcaz_register as an empty DataFrame at the start
    mcaz_register = pd.DataFrame()       

    # Initialize the variable to None or an empty list
    selected_generic_names = []   

    # Sidebar for navigation
    menu = ['Data Overview', 'Market Analysis', 'Manufacturer Analysis', 'FDA Orange Book Analysis', 
            'Applicant Analysis', 'Drugs@FDA Analysis','Patient-flow Forecast', 'Drug Classification Analysis', 
            'Drugs with no Competition', 'Top Pharma Companies Sales', 'FDA Drug Establishment Sites', 
            'FDA NME & New Biologic Approvals', 'EMA FDA Health Canada Approvals 2023']
    choice = st.sidebar.radio("Menu", menu)
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your MCAZ Register CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the data once and use it throughout
        data = load_data(uploaded_file)
        # Normalize column names immediately after loading
        data.columns = [str(col).strip() for col in data.columns]
        
        # Data Overview
        if choice == 'Data Overview':
            st.subheader('Data Overview')

            # Use the loaded and normalized 'data' directly
            mcaz_register = data.copy()
            st.session_state['mcaz_register'] = mcaz_register

            # Required columns
            required_columns_overview = [
                "Trade Name", "Generic Name", "Registration No", "Date Registered",
                "Expiry Date", "Form", "Categories for Distribution", "Strength",
                "Manufacturers", "Applicant Name", "Principal Name"
            ]

            # Check if all required columns exist in the data
            missing_columns = [col for col in required_columns_overview if col not in data.columns]
            if missing_columns:
                st.error(f"The following required columns are missing from the uploaded data: {', '.join(missing_columns)}")
                # Use a conditional block to stop further processing
                # At this point, you've informed the user what's wrong. You can prompt them to re-upload or fix the file.
                st.info("Please upload a file that includes all the required columns.")
            else:
            # Proceed with processing that depends on the presence of required columns
          
                # Manufacturer Filter
                # Check if 'Manufacturers' column exists in the data
                if 'Manufacturers' in data.columns:
                    manufacturer_options = ['All Manufacturers'] + sorted(data['Manufacturers'].dropna().unique().tolist())
                    selected_manufacturer = st.selectbox('Select Manufacturer', manufacturer_options, index=0)
                else:
                    st.error("The 'Manufacturers' column is missing from the uploaded data.")

                data.columns = [str(col).strip() for col in data.columns]
                product_options = ['All Products'] + sorted(data['Generic Name'].dropna().unique().tolist())
                selected_product = st.selectbox('Select Generic Name', product_options, index=0)

                # Form Filter
                form_options = ['All Forms'] + sorted(data['Form'].dropna().unique().tolist())
                selected_form = st.selectbox('Select Form', form_options, index=0)

                # Principal Filter
                principal_options = ['All Principal'] + sorted(data['Principal Name'].dropna().unique().tolist())
                selected_principal = st.selectbox('Select Principal Name', principal_options, index=0)

                # Categories of Distribution Filter
                category_options = ['All Categories of Distribution'] + sorted(data['Categories for Distribution'].dropna().unique().tolist())
                selected_category = st.selectbox('Select Category of Distribution', category_options, index=0)

                # Applicant Filter
                applicant_options = ['All Applicants'] + sorted(data['Applicant Name'].dropna().unique().tolist())
                selected_applicant = st.selectbox('Select Applicant Name', applicant_options, index=0)

                # Sort Order Filter for Generic Name
                sort_order_generic_options = ['Ascending', 'Descending']
                selected_sort_order_generic = st.selectbox('Sort by Generic Name', sort_order_generic_options)

                # Sort Order Filter for Strength
                sort_order_strength_options = ['Ascending', 'Descending']
                selected_sort_order_strength = st.selectbox('Sort by Strength', sort_order_strength_options)


                # Filtering the data based on selections
                filtered_data = data
                if selected_manufacturer != 'All Manufacturers':
                    filtered_data = filtered_data[filtered_data['Manufacturers'] == selected_manufacturer]
                if selected_product != 'All Products':
                    filtered_data = filtered_data[filtered_data['Generic Name'] == selected_product]
                if selected_form != 'All Forms':
                    filtered_data = filtered_data[filtered_data['Form'] == selected_form]
                if selected_principal != 'All Principal':
                    filtered_data = filtered_data[filtered_data['Principal Name'] == selected_principal]
                if selected_category != 'All Categories of Distribution':
                    filtered_data = filtered_data[filtered_data['Categories for Distribution'] == selected_category]
                if selected_applicant != 'All Applicants':
                    filtered_data = filtered_data[filtered_data['Applicant Name'] == selected_applicant]

                # Apply sort order for Generic Name and then Strength
                if selected_sort_order_generic == 'Descending':
                    filtered_data = filtered_data.sort_values(by=['Generic Name', 'Strength'], ascending=[False, selected_sort_order_strength == 'Ascending'])
                else:
                    filtered_data = filtered_data.sort_values(by=['Generic Name', 'Strength'], ascending=[True, selected_sort_order_strength == 'Ascending'])

                # Display the filtered dataframe
                st.write("Filtered Data:")
                st.dataframe(filtered_data)
                st.write(f"Filtered data count: {len(filtered_data)}")

                # Download Dataframe
                csv = convert_df_to_csv(filtered_data)
                st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_data.csv', mime='text/csv')
            
            # Start of the Streamlit UI layout
            st.subheader("Data Processing with Fuzzy Matching and ATC Code Extraction")

            # Choose the type of medicine
            medicine_type = st.radio("Select Medicine Type", ["Human Medicine", "Veterinary Medicine"])

            # Initialize session state for fuzzy matching data and ATC level data
            if 'fuzzy_matched_data' not in st.session_state:
                st.session_state.fuzzy_matched_data = pd.DataFrame()
            if 'atc_level_data' not in st.session_state:
                st.session_state.atc_level_data = pd.DataFrame()

            mcaz_register_file = st.file_uploader("Upload MCAZ Register File", type=['csv'], key="mcaz_register_uploader")
            atc_index_file = st.file_uploader(f"Upload {'Human' if medicine_type == 'Human Medicine' else 'Veterinary'} ATC Index File", type=['csv'], key="atc_index_uploader")

            # Initialize or reset session state variables
            if 'processed_rows' not in st.session_state:
                st.session_state.processed_rows = 0
            if 'resume_processing' not in st.session_state:
                st.session_state.resume_processing = False

            # Verify files are uploaded and have the required columns before processing
            if mcaz_register_file and atc_index_file:
                mcaz_register = load_file(mcaz_register_file)
                atc_index = load_file(atc_index_file)
                mcaz_register = init_columns(mcaz_register)

                # Required columns for MCAZ register and ATC Index file
                required_mcaz_columns = ['Generic Name', 'Strength', 'Form', 'Categories for Distribution', 'Manufacturers', 'Applicant Name','Principal Name']
                required_atc_columns = ['ATCCode', 'Name']

                # Check if uploaded files contain the required columns
                if not all(column in mcaz_register.columns for column in required_mcaz_columns):
                    st.error("MCAZ Register file is missing one or more required columns.")
                elif not all(column in atc_index.columns for column in required_atc_columns):
                    st.error("ATC Index file is missing one or more required columns.")
                else:
                    # Proceed with processing if all required columns are present
                    if st.button("Start/Resume Processing"):
                        st.session_state.resume_processing = True
                        process_data(mcaz_register, atc_index, medicine_type)
            else:
                st.error("Please upload both MCAZ Register and ATC Index files to proceed.")

            # Reset and clear session state
            if st.button("Reset Processing"):
                for key in ['processed_rows', 'resume_processing', 'start_time', 'end_time', 'fuzzy_matched_data', 'atc_level_data']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
                            
            # Display the processed data only if it exists in session state
            if 'fuzzy_matched_data' in st.session_state and not st.session_state.fuzzy_matched_data.empty:
                st.write("Updated MCAZ Register with Fuzzy Matching and ATC Codes:")
                st.dataframe(st.session_state.fuzzy_matched_data)
                # ... [Rest of your code]
                  # Perform further processing on the mcaz_register DataFrame
                # Make sure to use the DataFrame from session state
                mcaz_register = st.session_state.fuzzy_matched_data
            
                # Check if the DataFrame has the required columns before accessing them
                required_columns = ['Generic Name', 'Strength', 'Form', 'Categories for Distribution', 'Manufacturers', 
                                    'Principal Name', 'Best Match Name', 'Match Score', 'ATCCode']
                if all(col in mcaz_register.columns for col in required_columns):
                    mcaz_register = mcaz_register[required_columns]
                    # ... [Any further operations on mcaz_register]
                else:
                    st.error("Missing required columns in the dataset.")
            
            # Download file
            csv = convert_df_to_csv(mcaz_register)
            if csv is not None:
                # Proceed with operations that use 'csv'
                st.download_button(label="Download MCAZ Register as CSV", data=csv, file_name='mcaz_register_with_atc_codes.csv', mime='text/csv', key='download_mcaz_withcodes')

            else:
                # Handle the case where 'csv' is None, e.g., display a message or take alternative action
                print("No data available to convert to CSV")
                
            if mcaz_register is not None:
                try:
                    mcaz_register = mcaz_register[['Generic Name', 'Strength', 'Form', 'Categories for Distribution', 'Manufacturers', 'Principal Name', 'Best Match Name', 'Match Score', 'ATCCode']]

                    # Convert all strings in the DataFrame to uppercase
                    for column in mcaz_register.columns:
                        mcaz_register[column] = mcaz_register[column].map(lambda x: x.upper() if isinstance(x, str) else x)

                    # Assuming extract_atc_levels_human and extract_atc_levels_veterinary are defined
                    extract_atc_levels = extract_atc_levels_human if medicine_type == 'Human Medicine' else extract_atc_levels_veterinary

                    # Apply the function to each ATC code in the DataFrame
                    atc_data = mcaz_register['ATCCode'].apply(lambda x: pd.Series(extract_atc_levels(x)))
                    atc_data.columns = ['ATCLevelOneCode', 'ATCLevelTwoCode', 'ATCLevelThreeCode', 'ATCLevelFourCode']
                    mcaz_register = pd.concat([mcaz_register, atc_data], axis=1)

                    st.session_state.atc_level_data = mcaz_register

                    if not st.session_state.atc_level_data.empty:
                        st.write("Updated MCAZ Register with ATC Level Codes:")
                        st.dataframe(st.session_state.atc_level_data)

                        # Download file
                        csv = convert_df_to_csv(st.session_state.atc_level_data)
                        st.download_button(label="Download MCAZ Register as CSV", data=csv, file_name='mcaz_register_with_ATC_Level_Codes.csv', mime='text/csv', key='download_updated_register')
                except KeyError as e:
                    print(f"Column not found in DataFrame: {e}")
            else:
                print("mcaz_register is None. Please check data loading and processing steps.")

            # Streamlit UI layout for ATC Code Description Integration and Filtering
            st.subheader("ATC Code Description Integration and Filtering")
            
            # Initialize session state for check_passed
            if 'check_passed' not in st.session_state:
                st.session_state['check_passed'] = False

            # Initialize variables for ATC data and filter variables
            atc_one = atc_two = atc_three = atc_four = None
            atc_one_desc = atc_two_desc = atc_three_desc = atc_four_desc = selected_generic_names = []

            # Required columns for each ATC level
            required_columns_atc_one = ['ATCLevelOneCode', 'ATCLevelOneDescript']
            required_columns_atc_two = ['ATCLevelTwoCode', 'ATCLevelTwoDescript']
            required_columns_atc_three = ['ATCLevelThreeCode', 'ATCLevelThreeDescript']
            required_columns_atc_four = ['ATCLevelFourCode', 'Chemical Subgroup']

            # File uploaders for ATC level description files
            atc_one_file = st.file_uploader("Upload ATC Level One Description File", type=['csv'], key="atc_one_uploader_one")
            atc_two_file = st.file_uploader("Upload ATC Level Two Description File", type=['csv'], key="atc_two_uploader_two")
            atc_three_file = st.file_uploader("Upload ATC Level Three Description File", type=['csv'], key="atc_three_uploader_three")
            atc_four_file = st.file_uploader("Upload ATC Level Four Description File", type=['csv'], key="atc_four_uploader_four")

            # Button to trigger the check operation
            check_data = st.button("Check Required Columns")

            if check_data:
                # Reset the check_passed flag
                st.session_state['check_passed'] = True

                # Load ATC description files if uploaded
                atc_one = safe_load_csv(atc_one_file) if atc_one_file else None
                atc_two = safe_load_csv(atc_two_file) if atc_two_file else None
                atc_three = safe_load_csv(atc_three_file) if atc_three_file else None
                atc_four = safe_load_csv(atc_four_file) if atc_four_file else None

                # Check for required columns in each ATC level description file
                check_required_columns(atc_one, required_columns_atc_one, "One")
                check_required_columns(atc_two, required_columns_atc_two, "Two")
                check_required_columns(atc_three, required_columns_atc_three, "Three")
                check_required_columns(atc_four, required_columns_atc_four, "Four")
                
            else:
                st.info("Please upload ATC level description files and press 'Check Required Columns'.")

            # Button to trigger the merge operation
            merge_data = st.button("Merge Data")

            if merge_data:
                if 'fuzzy_matched_data' in st.session_state and not st.session_state.fuzzy_matched_data.empty:
                    if st.session_state['check_passed']:  # Check if all required columns are present
                        # Your merging logic here...
                        # Load ATC description files if uploaded
                        atc_one = safe_load_csv(atc_one_file) if atc_one_file else None
                        atc_two = safe_load_csv(atc_two_file) if atc_two_file else None
                        atc_three = safe_load_csv(atc_three_file) if atc_three_file else None
                        atc_four = safe_load_csv(atc_four_file) if atc_four_file else None
                        
                        # Retrieve the fuzzy_matched_data from session state
                        mcaz_register =  st.session_state.atc_level_data

                        # Merge with ATC level descriptions
                        with st.spinner('Merging data with ATC level descriptions...'):
                            if atc_one is not None and 'ATCLevelOneCode' in mcaz_register.columns:
                                mcaz_register = mcaz_register.merge(atc_one, on='ATCLevelOneCode', how='left')
                            if atc_two is not None and 'ATCLevelTwoCode' in mcaz_register.columns:
                                mcaz_register = mcaz_register.merge(atc_two, on='ATCLevelTwoCode', how='left')
                            if atc_three is not None and 'ATCLevelThreeCode' in mcaz_register.columns:
                                mcaz_register = mcaz_register.merge(atc_three, on='ATCLevelThreeCode', how='left')
                            if atc_four is not None and 'ATCLevelFourCode' in mcaz_register.columns:
                                mcaz_register = mcaz_register.merge(atc_four, on='ATCLevelFourCode', how='left')

                        # Correctly update the session state with the merged data
                        st.session_state['mcaz_with_ATCCodeDescription'] = mcaz_register
                        st.success("Data merged with ATC level descriptions.")

                        # Display the merged dataframe
                        if not mcaz_register.empty:
                            st.write("Merged Data:")
                            st.dataframe(mcaz_register)
                        else:
                            st.write("No data to display after merging.")

                        st.success("Data merged with ATC level descriptions.")
                    else:
                        st.error("Cannot merge data. Please ensure all required columns are present and try again.")
                else:
                    st.warning("Please complete the fuzzy matching process and ensure ATC level description files are uploaded.")

            # Download file
            csv = convert_df_to_csv(mcaz_register)
            if csv is not None:
                # Proceed with operations that use 'csv'
                st.download_button(label="Download MCAZ Register as CSV", data=csv, file_name='mcaz_register_with_atc_description.csv', mime='text/csv', key='download_mcaz_withatcdescription')

            else:
                # Handle the case where 'csv' is None, e.g., display a message or take alternative action
                print("No data available to convert to CSV")
                
            # Filter options presented to the user
            filter_options = ["None", "ATCLevelOneDescript", "ATCLevelTwoDescript", "ATCLevelThreeDescript", "Chemical Subgroup", "Generic Name"]
            selected_filter = st.radio("Select a filter", filter_options)

            # Check if 'fda_with_ATCCodeDescription' is in session state and is not empty
            if 'mcaz_with_ATCCodeDescription' in st.session_state and not st.session_state['mcaz_with_ATCCodeDescription'].empty:
                # Condition to handle filtering
                if selected_filter != "None":
                    # Convert all values in the selected filter column to string, get unique values, and sort
                    filter_values = sorted(st.session_state['mcaz_with_ATCCodeDescription'][selected_filter].astype(str).unique())
                    selected_values = st.multiselect(f"Select {selected_filter}", filter_values)

                    # Initialize filtered_data with the full dataset to handle scope
                    filtered_data = st.session_state['mcaz_with_ATCCodeDescription'].copy()

                    if selected_values:
                        # Apply filter based on selected values
                        filtered_data = filtered_data[filtered_data[selected_filter].astype(str).isin(selected_values)]
                        st.write(f"Filtered Data by {selected_filter}:")
                    else:
                        st.write("Displaying unfiltered data (no specific filter values selected):")

                    # Display filtered data and count
                    st.dataframe(filtered_data)
                    st.write(f"Filtered data count: {len(filtered_data)}")

                    # Offer CSV download for filtered data
                    csv = convert_df_to_csv(filtered_data)
                    st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='mcaz_register_filtered.csv', mime='text/csv', key='download_filtered')
                else:
                    st.write("No filter selected. Displaying unfiltered data:")
                    st.dataframe(st.session_state['mcaz_with_ATCCodeDescription'])
                    st.write(f"Total data count: {len(st.session_state['mcaz_with_ATCCodeDescription'])}")

                    # Optionally, offer download of unfiltered data
                    csv = convert_df_to_csv(st.session_state['mcaz_with_ATCCodeDescription'])
                    st.download_button(label="Download Unfiltered Data as CSV", data=csv, file_name='mcaz_register_unfiltered.csv', mime='text/csv', key='download_unfiltered')
            else:
                # Handle case where the data isn't available or hasn't been loaded
                st.write("Data not available. Please ensure data is loaded and processed.")
           
            # Streamlit UI layout for Data Filtering Based on User Type and Selected Filter
            st.subheader("Data Filtering Based on User Type and Selected Filter")

            # Medicine type selection
            medicine_type_options = ["Select Medicine Type", "Human Medicine", "Veterinary Medicine"]
            selected_medicine_type = st.selectbox("Select Medicine Type", medicine_type_options)

            # Initialize an empty DataFrame for mcaz_register to handle its scope outside the if condition
            mcaz_register = pd.DataFrame()

            # Only proceed with user type filtering if "Human Medicine" is selected and data has been merged
            if selected_medicine_type == "Human Medicine":
                user_type_options = ["None", "Local Manufacturer", "Importer"]
                user_type = st.radio("Select User Type", user_type_options)

                prohibited_file = st.file_uploader("Upload Prohibited Generics List With Dosage Forms", type=['csv'])

                if prohibited_file is not None:
                    # Attempt to read the uploaded file for column verification
                    try:
                        temp_df = pd.read_csv(prohibited_file)
                        # Reset the file pointer after reading
                        prohibited_file.seek(0)
                        required_columns = ['Generic Name', 'Form']
                        check_passed, missing_columns = check_prohibited_file_columns(temp_df, required_columns)
                        if check_passed:
                            st.success("Uploaded file contains all required columns.")
                        else:
                            st.error(f"Uploaded file is missing required columns: {', '.join(missing_columns)}")
                            # Skip further processing if required columns are missing
                            prohibited_file = None
                    except Exception as e:
                        st.error(f"An error occurred while processing the file: {str(e)}")
                        prohibited_file = None

                filter_options = ["None", "ATCLevelOneDescript", "ATCLevelTwoDescript", 
                                  "ATCLevelThreeDescript", "Chemical Subgroup", "Generic Name"]
                selected_filter = st.radio("Select an additional filter", filter_options)

                if 'mcaz_with_ATCCodeDescription' in st.session_state and not st.session_state['mcaz_with_ATCCodeDescription'].empty:
                    mcaz_register = st.session_state['mcaz_with_ATCCodeDescription']

                    if prohibited_file and user_type != "None":
                        prohibited_generics = load_and_process_prohibited_generics(prohibited_file)
                        mcaz_register = filter_data_for_user(user_type, mcaz_register, prohibited_generics)
                        mcaz_register = mcaz_register.drop_duplicates()

                    if selected_filter != "None":
                        filter_values = sorted(mcaz_register[selected_filter].astype(str).unique())
                        selected_values = st.multiselect(f"Select {selected_filter}", filter_values, key="valid")

                        if selected_values:
                            mcaz_register = mcaz_register[mcaz_register[selected_filter].astype(str).isin(selected_values)]

                    st.write("Filtered Data:")
                    st.dataframe(mcaz_register)
                    st.write(f"Filtered data count: {len(mcaz_register)}")

                    csv = convert_df_to_csv(mcaz_register)
                    if csv is not None:
                        st.download_button(label="Download MCAZ Register as CSV", data=csv, file_name='mcaz_register_prohibited_medicine.csv', mime='text/csv', key='download_mcaz_filtered')
                else:
                    st.error("Data not available in the session state or no data to display after filtering.")
            else:
                st.error("Select 'Human Medicine' to access user type based data filtering.")
       
        # Market Analysis
        elif choice == 'Market Analysis':
            st.subheader('Market Analysis')
            # Manufacturer selection logic
            # Placeholder for manufacturer filtering (if not yet implemented)
            all_manufacturers = ['All Manufacturers'] + sorted(data['Manufacturers'].dropna().unique().tolist())
            selected_manufacturer = st.selectbox('Select Manufacturer', all_manufacturers, index=0, key="manufacturer_select")
            if selected_manufacturer == 'All Manufacturers':
                manufacturer_filtered_data = data
            else:
                manufacturer_filtered_data = data[data['Manufacturers'] == selected_manufacturer]

            # Form selection logic
            all_forms = ['All Forms'] + sorted(data['Form'].dropna().unique().tolist())
            selected_forms = st.multiselect('Select Forms', all_forms, default='All Forms')
            if 'All Forms' in selected_forms or not selected_forms:
                form_filtered_data = manufacturer_filtered_data
            else:
                form_filtered_data = manufacturer_filtered_data[manufacturer_filtered_data['Form'].isin(selected_forms)]

            # Visualization (e.g., distribution of drug forms)
            form_counts = form_filtered_data['Form'].value_counts()
            st.bar_chart(form_counts)

            # Streamlit UI components for "Generic Name Count"
            st.subheader('Generic Name Count')

            # Load the data using the existing function
            data = load_data(uploaded_file)

            # Count unique generic names and their frequencies
            unique_generic_name_count = data['Generic Name'].nunique()
            generic_name_counts = data['Generic Name'].value_counts()

            # Display the counts
            st.write(f"Total unique generic names: {unique_generic_name_count}")
            st.write("Top Generic Names by Count:")
            st.dataframe(generic_name_counts)

            # Download button for unique product count
            if not generic_name_counts.empty:
                csv = generic_name_counts.to_csv(index=False)
                st.download_button("Download Generic Name Data", csv, "generic_data.csv", "text/csv", key='download-unique-generic')

            # Unique generic name count
            st.subheader('Unique Generic Name Count')

            # Count unique generic names and their frequencies
            generic_name_counts = data['Generic Name'].value_counts()

            # Filter options
            filter_options = ['3 or less', '4', '5', '6', '7 or more']
            selected_filter = st.selectbox('Select count filter:', filter_options, index=0)

            # Apply filter
            if selected_filter == '3 or less':
                filtered_counts = generic_name_counts[generic_name_counts <= 3]
            elif selected_filter == '7 or more':
                filtered_counts = generic_name_counts[generic_name_counts >= 7]
            else:
                count_value = int(selected_filter)
                filtered_counts = generic_name_counts[generic_name_counts == count_value]

            # Display the counts
            st.write(f"Generic Names with {selected_filter} counts:")
            st.dataframe(filtered_counts)

            # Download button for generic name counts
            if not filtered_counts.empty:
                csv = filtered_counts.to_csv(index=False)
                st.download_button("Download Generic Name Counts", csv, "generic_name_counts.csv", "text/csv", key='download-generic-name-count')


            # Show unique products
            st.subheader('Unique Products Count')

            # Load the data using the existing function
            data = load_data(uploaded_file)

            # Create a new column combining 'Generic Name', 'Strength', and 'Form'
            data['Combined'] = data['Generic Name'] + " - " + data['Strength'].astype(str) + " - " + data['Form']

            # Product Filter
            product_options = ['All Products'] + sorted(data['Combined'].dropna().unique().tolist())
            selected_product = st.selectbox('Select Product', product_options, index=0)

            # Filter the data based on the selected product
            if selected_product != 'All Products':
                filtered_data = data[data['Combined'] == selected_product]
            else:
                filtered_data = data

            # Count unique products based on 'Combined'
            unique_product_count = filtered_data['Combined'].nunique()

            # Display the count
            st.write(f"Total unique products (by Generic Name, Strength, and Form): {unique_product_count}")

            # Value count filter options
            count_filter_options = ['3 or less', '4', '5', '6', '7 or more']
            selected_count_filter = st.selectbox('Select value count filter:', count_filter_options)

            # Filter and display the list of unique products with their count
            if st.checkbox("Show List of Unique Products", value=True):
                unique_products_counts = filtered_data['Combined'].value_counts()

                if selected_count_filter == '3 or less':
                    filtered_counts = unique_products_counts[unique_products_counts <= 3]
                elif selected_count_filter == '4':
                    filtered_counts = unique_products_counts[unique_products_counts == 4]
                elif selected_count_filter == '5':
                    filtered_counts = unique_products_counts[unique_products_counts == 5]
                elif selected_count_filter == '6':
                    filtered_counts = unique_products_counts[unique_products_counts == 6]
                elif selected_count_filter == '7 or more':
                    filtered_counts = unique_products_counts[unique_products_counts >= 7]

                st.write(filtered_counts)

                # Download button for unique product count
                if not filtered_counts.empty:
                    csv = filtered_counts.to_csv(index=False)
                    st.download_button("Download Unique Products Data", csv, "unique_products_data.csv", "text/csv", key='download-unique-product')
            
            # Streamlit widget to upload the Prohibited Medicines file
            uploaded_prohibited_file = st.file_uploader("Upload Prohibited Medicines file With Strength and Dosage Form", type=['csv'])
            
            # Required columns
            required_columns = ['Generic Name', 'Strength', 'Form']

            # Check for required columns if a file is uploaded
            file_check_passed, missing_columns = check_required_columns_in_file(uploaded_prohibited_file, required_columns)

            if file_check_passed is True:
                st.success("Uploaded file contains all required columns.")
            elif file_check_passed is False:
                st.error(f"Uploaded file is missing required columns: {', '.join(missing_columns)}")
            elif file_check_passed is None and uploaded_prohibited_file is not None:
                st.warning("Please upload a file to proceed.")

            # Assuming you have a variable to capture the user type
            user_type = st.selectbox('Select User Type', ['None', 'Importer', 'Local Manufacturer'])
            
            if uploaded_prohibited_file is not None:
                uploaded_prohibited_file.seek(0)  # Reset file pointer to the start
                try:
                    # Attempt to load the Prohibited Medicines file
                    prohibited_medicines_df = pd.read_csv(uploaded_prohibited_file)

                    # Ensure the DataFrame is not empty by checking if it has columns
                    if prohibited_medicines_df.empty:
                        st.error("Uploaded file is empty or does not contain any data.")
                    else:
                        # Convert column names to uppercase to match the MCAZ Register
                        prohibited_medicines_df.columns = prohibited_medicines_df.columns.str.upper()
                
                        # Combine 'GENERIC NAME', 'STRENGTH', and 'FORM' to match the 'Combined' format in your data dataframe
                        prohibited_medicines_df['COMBINED'] = prohibited_medicines_df['GENERIC NAME'] + " - " + prohibited_medicines_df['STRENGTH'].astype(str) + " - " + prohibited_medicines_df['FORM']

                        # Example to integrate the prohibited medicines filter based on the user type
                        if user_type == 'Importer':
                            # Assuming 'filtered_counts' is already defined in your code with the counts of unique products
                            # First, ensure 'filtered_counts' is in a format that can be filtered (e.g., a DataFrame)

                            # Exclude prohibited medicines for importers
                            prohibited_list = prohibited_medicines_df['COMBINED'].tolist()
                            filtered_counts = filtered_counts[~filtered_counts.index.isin(prohibited_list)]

                            # Display the filtered 'filtered_counts' DataFrame
                            st.write(filtered_counts)
                            
                            # Download button for unique product count
                            if not filtered_counts.empty:
                                csv = filtered_counts.to_csv(index=False)
                                st.download_button("Download Unique Products Data", csv, "unique_products_importer_data.csv", "text/csv", key='download-unique-product_importer')

                    
                except Exception as e:
                    # This will catch all exceptions, including any related to empty data or parsing issues
                    st.error(f"An error occurred while processing the file: {str(e)}")
                   
        # Manufacturer Analysis
        elif choice == 'Manufacturer Analysis':
            st.subheader('Manufacturer Analysis')

            # Ensure all manufacturers are strings and handle NaN values
            all_manufacturers = data['Manufacturers'].dropna().unique()
            all_manufacturers = [str(manufacturer) for manufacturer in all_manufacturers]
            all_manufacturers.sort()

            # Adding 'All Manufacturers' option
            manufacturers_options = ['All Manufacturers'] + all_manufacturers
            selected_manufacturer = st.selectbox('Select Manufacturer', manufacturers_options, index=0)

            # Filtering data based on the selected manufacturer
            if selected_manufacturer == 'All Manufacturers':
                filtered_data = data
            else:
                filtered_data = data[data['Manufacturers'] == selected_manufacturer]

            # Convert 'Date Registered' to datetime
            filtered_data['Date Registered'] = pd.to_datetime(filtered_data['Date Registered'])
            
           # Yearly trend analysis
            yearly_trend = filtered_data['Date Registered'].dt.year.value_counts().sort_index()
            st.line_chart(yearly_trend)

            # Main submodule for Principal Product Count
            st.subheader("Principal Product Count")

            # Data
            filtered_counts = pd.DataFrame()
            data = load_data(uploaded_file)

            if not data.empty:
                # Apply fuzzy matching to 'Principal Name'
                data['Fuzzy Matched Principal'] = fuzzy_match_names(data['Principal Name'])

                # Count products by fuzzy matched principal name
                principal_counts = (data.groupby('Fuzzy Matched Principal')['Generic Name']
                                    .count()
                                    .reset_index()
                                    .rename(columns={'Fuzzy Matched Principal': 'Principal Name', 'Generic Name': 'Generic Name Count'})
                                    .sort_values(by='Generic Name Count', ascending=False))

                st.write("Product Count by Principal:")
                st.dataframe(principal_counts)

                # Display the total count of products
                total_products = len(data)
                st.write(f"Total Count of Products: {total_products}")

                # Convert the complete DataFrame to CSV
                csv_data = convert_df_to_csv(principal_counts)
                st.download_button(
                    label="Download data as CSV",
                    data=csv_data,
                    file_name='principal_product_count.csv',
                    mime='text/csv',
                )
            else:
                st.write("No data available.")

            # Anatomial Main Group
            st.subheader("Anatomcal Main Group Count")

            if 'mcaz_with_ATCCodeDescription' in st.session_state and not st.session_state['mcaz_with_ATCCodeDescription'].empty:
                mcaz_register = st.session_state['mcaz_with_ATCCodeDescription']

                # Remove complete duplicates
                mcaz_register = mcaz_register.drop_duplicates()

                if not mcaz_register.empty:
                    # Convert 'Principal Name' to string and handle NaN values
                    mcaz_register['Principal Name'] = mcaz_register['Principal Name'].fillna('Unknown').astype(str)

                    # Add "None" option and select Principal Name
                    principal_options = ['None'] + sorted(mcaz_register['Principal Name'].unique())
                    selected_principal = st.selectbox("Select Principal Name", principal_options)

                    # Choose sort order
                    sort_order = st.radio("Select Sort Order", ["Ascending", "Descending"])

                    if selected_principal != "None":
                        # Filter data based on selected principal
                        filtered_data = mcaz_register[mcaz_register['Principal Name'] == selected_principal]
                    else:
                        # If "None" is selected, use the entire dataset
                        filtered_data = mcaz_register

                    # Group by 'ATC Level One Description' and count unique 'Generic Name'
                    # Sort based on the selected sort order
                    if sort_order == "Descending":
                        atc_classification_count = (filtered_data.groupby('ATCLevelOneDescript')['Generic Name']
                                                    .count()
                                                    .reset_index()
                                                    .rename(columns={'Generic Name': 'GenericNameCount'})
                                                    .sort_values(by='GenericNameCount', ascending=False))
                    else:
                        atc_classification_count = (filtered_data.groupby('ATCLevelOneDescript')['Generic Name']
                                                    .count()
                                                    .reset_index()
                                                    .rename(columns={'Generic Name': 'GenericNameCount'})
                                                    .sort_values(by='GenericNameCount', ascending=True))

                    st.write(f"Count of Generic Name by ATC Level One Description (sorted {sort_order}):")
                    st.dataframe(atc_classification_count)

                    # Calculate the total count of products across all ATC Level One Descriptions
                    # Ensure you're summing only the 'GenericNameCount' column
                    total_product_count = atc_classification_count['GenericNameCount'].sum()
                    st.write(f"Total Count of Products (Across All Groups): {total_product_count}")

                    # Convert the complete DataFrame to CSV
                    csv_data = convert_df_to_csv(atc_classification_count)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv_data,
                        file_name='atc_classification_one_count.csv',
                        mime='text/csv',
                    )

                else:
                    st.write("No data available.")
            else:
                st.write("ATC Code Description data is not available.")

            # Pharmacological Group
            st.subheader("Pharmacological Group Count")

            if 'mcaz_with_ATCCodeDescription' in st.session_state and not st.session_state['mcaz_with_ATCCodeDescription'].empty:
                mcaz_register = st.session_state['mcaz_with_ATCCodeDescription']

                # Remove complete duplicates
                mcaz_register = mcaz_register.drop_duplicates()

                if not mcaz_register.empty:
                    # Convert 'Principal Name' to string and handle NaN values
                    mcaz_register['Principal Name'] = mcaz_register['Principal Name'].fillna('Unknown').astype(str)

                    # Add "None" option and select Principal Name
                    principal_options = ['None'] + sorted(mcaz_register['Principal Name'].unique())
                    selected_principal_1 = st.selectbox("Select Principal Name", principal_options, key = "principal_selection_1")

                    # Choose sort order
                    sort_order_1 = st.radio("Select Sort Order", ["Ascending", "Descending"], key = "sort_order_selection_1")

                    if selected_principal_1 != "None":
                        # Filter data based on selected principal
                        filtered_data = mcaz_register[mcaz_register['Principal Name'] == selected_principal_1]
                    else:
                        # If "None" is selected, use the entire dataset
                        filtered_data = mcaz_register

                    # Group by 'ATC Level One Description' and count unique 'Generic Name'
                    # Sort based on the selected sort order
                    if sort_order_1 == "Descending":
                        atc_classification_count = (filtered_data.groupby('ATCLevelTwoDescript')['Generic Name']
                                                    .count()
                                                    .reset_index()
                                                    .rename(columns={'Generic Name': 'GenericNameCount'})
                                                    .sort_values(by='GenericNameCount', ascending=False))
                    else:
                        atc_classification_count = (filtered_data.groupby('ATCLevelTwoDescript')['Generic Name']
                                                    .count()
                                                    .reset_index()
                                                    .rename(columns={'Generic Name': 'GenericNameCount'})
                                                    .sort_values(by='GenericNameCount', ascending=True))

                    st.write(f"Count of Generic Name by ATC Level Two Description (sorted {sort_order_1}):")
                    st.dataframe(atc_classification_count)

                    # Calculate the total count of products across all ATC Level One Descriptions
                    # Ensure you're summing only the 'GenericNameCount' column
                    total_product_count = atc_classification_count['GenericNameCount'].sum()
                    st.write(f"Total Count of Products (Across All Groups): {total_product_count}")

                    # Convert the complete DataFrame to CSV
                    csv_data = convert_df_to_csv(atc_classification_count)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv_data,
                        file_name='atc_classification_two_count.csv',
                        mime='text/csv', key = "pharmacology",
                    )

                else:
                    st.write("No data available.")
            else:
                st.write("ATC Code Description data is not available.")

            # Therapuetic Group
            st.subheader("Therapeutic Group Count")

            if 'mcaz_with_ATCCodeDescription' in st.session_state and not st.session_state['mcaz_with_ATCCodeDescription'].empty:
                mcaz_register = st.session_state['mcaz_with_ATCCodeDescription']

                # Remove complete duplicates
                mcaz_register = mcaz_register.drop_duplicates()

                if not mcaz_register.empty:
                    # Convert 'Principal Name' to string and handle NaN values
                    mcaz_register['Principal Name'] = mcaz_register['Principal Name'].fillna('Unknown').astype(str)

                    # Add "None" option and select Principal Name
                    principal_options = ['None'] + sorted(mcaz_register['Principal Name'].unique())
                    selected_principal_2 = st.selectbox("Select Principal Name", principal_options, key = "principal_selection_2")

                    # Choose sort order
                    sort_order_2 = st.radio("Select Sort Order", ["Ascending", "Descending"], key = "sort_order_selection_2")

                    if selected_principal_2 != "None":
                        # Filter data based on selected principal
                        filtered_data = mcaz_register[mcaz_register['Principal Name'] == selected_principal_2]
                    else:
                        # If "None" is selected, use the entire dataset
                        filtered_data = mcaz_register

                    # Group by 'ATC Level One Description' and count unique 'Generic Name'
                    # Sort based on the selected sort order
                    if sort_order_2 == "Descending":
                        atc_classification_count = (filtered_data.groupby('ATCLevelThreeDescript')['Generic Name']
                                                    .count()
                                                    .reset_index()
                                                    .rename(columns={'Generic Name': 'GenericNameCount'})
                                                    .sort_values(by='GenericNameCount', ascending=False))
                    else:
                        atc_classification_count = (filtered_data.groupby('ATCLevelThreeDescript')['Generic Name']
                                                    .count()
                                                    .reset_index()
                                                    .rename(columns={'Generic Name': 'GenericNameCount'})
                                                    .sort_values(by='GenericNameCount', ascending=True))

                    st.write(f"Count of Generic Name by ATC Level Three Description (sorted {sort_order_1}):")
                    st.dataframe(atc_classification_count)

                    # Calculate the total count of products across all ATC Level One Descriptions
                    # Ensure you're summing only the 'GenericNameCount' column
                    total_product_count = atc_classification_count['GenericNameCount'].sum()
                    st.write(f"Total Count of Products (Across All Groups): {total_product_count}")

                    # Convert the complete DataFrame to CSV
                    csv_data = convert_df_to_csv(atc_classification_count)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv_data,
                        file_name='atc_classification_three_count.csv',
                        mime='text/csv', key = "therapeutic",
                    )

                else:
                    st.write("No data available.")
            else:
                st.write("ATC Code Description data is not available.")

        # FDA Orange Book Analysis
        elif choice == 'FDA Orange Book Analysis':
            # Check if the dataframes are already loaded in the session state
            if 'products_df' not in st.session_state or 'patent_df' not in st.session_state or 'exclusivity_df' not in st.session_state:
                st.title("FDA Orange Book Analysis")

                # File uploaders
                products_file = st.file_uploader("Upload the products.csv file", type=['csv'], key="products_uploader")
                patent_file = st.file_uploader("Upload the patent.csv file", type=['csv'], key="patent_uploader")
                exclusivity_file = st.file_uploader("Upload the exclusivity.csv file", type=['csv'], key="exclusivity_uploader")


                # Check for required columns in each dataframe
                products_columns_required = ['Ingredient', 'DF;Route', 'Trade_Name', 'Applicant', 'Strength']
                patent_columns_required = ['Appl_Type', 'Appl_No', 'Product_No', 'Patent_No', 'Patent_Expire_Date_Text', 'Drug_Substance_Flag', 'Drug_Product_Flag', 'Patent_Use_Code', 'Delist_Flag', 'Submission_Date']
                exclusivity_columns_required = ['Appl_Type', 'Appl_No', 'Product_No', 'Exclusivity_Code', 'Exclusivity_Date']
                
                # Initialize a flag to indicate all required files are uploaded
                all_files_uploaded = True

                # Attempt to load the products file
                if products_file:
                    products_df = load_data_orange(products_file)
                    products_check, products_missing = check_required_columns_orangebook(products_df, products_columns_required) if products_df is not None else (False, ["DataFrame is None"])
                    if not products_check:
                        st.error(f"Missing columns in products file: {', '.join(products_missing)}. Please upload a correct file.")
                else:
                    st.error("Products file is not uploaded.")
                    all_files_uploaded = False

                # Attempt to load the patent file
                if patent_file:
                    patent_df = load_data_orange(patent_file)
                    patent_check, patent_missing = check_required_columns_orangebook(patent_df, patent_columns_required) if patent_df is not None else (False, ["DataFrame is None"])
                    if not patent_check:
                        st.error(f"Missing columns in patent file: {', '.join(patent_missing)}. Please upload a correct file.")
                else:
                    st.error("Patent file is not uploaded.")
                    all_files_uploaded = False

                # Attempt to load the exclusivity file
                if exclusivity_file:
                    exclusivity_df = load_data_orange(exclusivity_file)
                    exclusivity_check, exclusivity_missing = check_required_columns_orangebook(exclusivity_df, exclusivity_columns_required) if exclusivity_df is not None else (False, ["DataFrame is None"])
                    if not exclusivity_check:
                        st.error(f"Missing columns in exclusivity file: {', '.join(exclusivity_missing)}. Please upload a correct file.")
                else:
                    st.error("Exclusivity file is not uploaded.")
                    all_files_uploaded = False

                # Only proceed if all files are correctly uploaded and loaded
                if all_files_uploaded:
                    # Store the dataframes in the session state or proceed with further processing
                    st.session_state.products_df = products_df if 'products_df' in locals() else None
                    st.session_state.patent_df = patent_df if 'patent_df' in locals() else None
                    st.session_state.exclusivity_df = exclusivity_df if 'exclusivity_df' in locals() else None
            
            # If the dataframes are in the session state, proceed with the analysis
            if 'products_df' in st.session_state and 'patent_df' in st.session_state and 'exclusivity_df' in st.session_state:
                # Perform the analysis using the dataframes from the session state
                merged_df = outer_join_dfs(st.session_state.products_df, st.session_state.patent_df, st.session_state.exclusivity_df, "Appl_No")

                # Remove duplicates
                merged_df = merged_df.drop_duplicates(subset=['Ingredient', 'DF;Route', 'Strength', 'Appl_No', 'Product_No_x', 'Patent_No'])

                # Remove records with no patents
                merged_df = merged_df.dropna(subset=['Patent_No'])

                # Remove records with "Type" equal to "DISCN"
                merged_df = merged_df[merged_df['Type'] != 'DISCN']

                # Filters
                ingredient = st.selectbox("Select Ingredient", ['None'] + sorted(merged_df['Ingredient'].dropna().unique().tolist()))
                df_route = st.selectbox("Select DF;Route", ['None'] + sorted(merged_df['DF;Route'].dropna().unique().tolist()))
                trade_name = st.selectbox("Select Trade Name", ['None'] + sorted(merged_df['Trade_Name'].dropna().unique().tolist()))
                applicant = st.selectbox("Select Applicant", ['None'] + sorted(merged_df['Applicant'].dropna().unique().tolist()))
                appl_type = st.selectbox("Select Appl Type", ['None'] + sorted(merged_df['Appl_Type'].dropna().unique().tolist()))
                type_filter = st.selectbox("Select Type", ['None'] + sorted(merged_df['Type'].dropna().unique().tolist()))
                rld = st.selectbox("Select Rerence Listed Drug", ['None'] + sorted(merged_df['RLD'].dropna().unique().tolist()))
                rs = st.selectbox("Select Reference Standard", ['None'] + sorted(merged_df['RS'].dropna().unique().tolist()))
                drug_product_flag = st.selectbox("Select Drug Product Flag", ['None'] + sorted(merged_df['Drug_Product_Flag'].dropna().unique().tolist()))
                drug_substance_flag = st.selectbox("Select Drug Substance Flag", ['None'] + sorted(merged_df['Drug_Substance_Flag'].dropna().unique().tolist()))

                # Apply filters
                if ingredient != "None": merged_df = filter_dataframe(merged_df, 'Ingredient', ingredient)
                if df_route != "None": merged_df = filter_dataframe(merged_df, 'DF;Route', df_route)
                if trade_name != "None": merged_df = filter_dataframe(merged_df, 'Trade_Name', trade_name)
                if applicant != "None": merged_df = filter_dataframe(merged_df, 'Applicant', applicant)
                if appl_type != "None": merged_df = filter_dataframe(merged_df, 'Appl_Type', appl_type)
                if type_filter != "None": merged_df = filter_dataframe(merged_df, 'Type', type_filter)
                if rld != "None": merged_df = filter_dataframe(merged_df, 'RLD', rld)
                if rs != "None": merged_df = filter_dataframe(merged_df, 'RS', rs)
                if drug_product_flag != "None": merged_df = filter_dataframe(merged_df, 'Drug_Product_Flag', drug_product_flag)
                if drug_substance_flag != "None": merged_df = filter_dataframe(merged_df, 'Drug_Substance_Flag', drug_substance_flag)

                # Display Dataframe
                st.write("Filtered FDA Orange Book Data:")
                st.dataframe(merged_df)

                # Display count of products
                product_count = len(merged_df)
                st.write(f"Number of Products: {product_count}")

                # Download as CSV
                csv = convert_df_to_csv(merged_df)
                st.download_button(label="Download data as CSV", data=csv, file_name='fda_orange_book_data.csv', mime='text/csv')
                
                # Assuming 'merged_df' is your merged DataFrame
                filtered_columns = [
                    'Ingredient', 'DF;Route', 'Strength', 'Trade_Name', 
                    'Applicant', 'Patent_No', 'Approval_Date', 
                    'Patent_Expire_Date_Text'
                ]

                # Select only the specified columns
                merged_df = merged_df[filtered_columns]
                
                # Convert 'Patent_No' column to string
                merged_df['Patent_No'] = merged_df['Patent_No'].astype(str)

                # Strip the trailing '.0' from 'Patent_No' column
                merged_df['Patent_No'] = merged_df['Patent_No'].str.rstrip('.0')
                
                # Check if an ingredient is selected
                if ingredient != "None":
                    # Google Patents base URL
                    google_patents_base_url = "https://patents.google.com/patent/"
                    # WIPO base URL
                    base_url = "https://patentscope.wipo.int/search/en/search.jsf?query="
                                                           
                    # Construct Google Patents link
                    merged_df['Google_Patents_Link'] = merged_df['Patent_No'].apply(lambda x: f'<a href="{google_patents_base_url}US{x}B2/en?oq={x}" target="_blank">US{x}B2 on Google Patents</a>')

                    # Construct WIPO link (assuming WIPO docId format is compatible with your Patent_No format; adjust as needed)
#                     merged_df['WIPO_Patent_Link'] = merged_df['Patent_No'].apply(lambda x: f'<a href="{base_url}{x}" target="_blank">{x}</a>')
                    merged_df['WIPO_Link'] = merged_df['Patent_No'].apply(construct_wipo_link)

                    
                    # Apply the function to the 'Patent_No' column to create a new 'Espacenet_Link' column
                    merged_df['Espacenet_Link'] = merged_df['Patent_No'].apply(construct_espacenet_link)

                    # Filter the DataFrame based on the selected ingredient
                    filtered_df = merged_df[merged_df['Ingredient'] == ingredient]

                    # HTML Style for left alignment of the link columns
                    left_align_style = "<style>td { text-align: left !important; }</style>"

                    # Display the DataFrame with hyperlinks for the selected ingredient
                    st.write(f"DataFrame with Hyperlinked Patent Numbers for Ingredient: {ingredient}")
                    st.markdown(left_align_style + filtered_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.write("Please select an ingredient to display detailed information.")
                    
                
            # Start of the Streamlit UI layout
            st.subheader("FDA Data Processing with Fuzzy Matching and ATC Code Extraction")

            medicine_type = st.radio("Select Medicine Type", ["Human Medicine", "Veterinary Medicine"])

            # Initialize or ensure session state variables are available
            if 'fuzzy_matched_data' not in st.session_state:
                st.session_state.fuzzy_matched_data = pd.DataFrame()
            if 'atc_level_data' not in st.session_state:
                st.session_state.atc_level_data = pd.DataFrame()
            if 'fda_register' not in st.session_state:
                st.session_state.fda_register = pd.DataFrame()

            fda_register_file = st.file_uploader("Upload FDA Register File", type=['csv'], key="fda_register_uploader")
            atc_index_file = st.file_uploader(f"Upload {'Human' if medicine_type == 'Human Medicine' else 'Veterinary'} ATC Index File", type=['csv'], key="atc_index_uploader")

            if 'processed_rows' not in st.session_state:
                st.session_state.processed_rows = 0
            if 'resume_processing' not in st.session_state:
                st.session_state.resume_processing = False

            if fda_register_file and atc_index_file:
                st.session_state.fda_register = load_file(fda_register_file)
                atc_index = load_file(atc_index_file)

                # Check for required columns in both files
                required_fda_columns = ['Ingredient', 'DF;Route', 'Strength', 'Trade_Name', 'Applicant']
                required_atc_columns = ['ATCCode', 'Name']

                if not all(column in st.session_state.fda_register.columns for column in required_fda_columns):
                    st.error("FDA Register file is missing one or more required columns.")
                elif not all(column in atc_index.columns for column in required_atc_columns):
                    st.error("ATC Index file is missing one or more required columns.")
                else:
                    st.session_state.fda_register = init_columns(st.session_state.fda_register)
                    
                    extract_atc_levels = extract_atc_levels_human if medicine_type == 'Human Medicine' else extract_atc_levels_veterinary

                    # Proceed with processing only if all required columns are present
                    if st.button("Start/Resume Processing"):
                        st.session_state.resume_processing = True
                        process_data_fda(st.session_state.fda_register, atc_index, extract_atc_levels)
            else:
                st.error("Please upload both FDA Register and ATC Index files to proceed.")

            if st.button("Reset Processing"):
                for key in ['processed_rows', 'resume_processing', 'start_time', 'end_time', 'fuzzy_matched_data', 'atc_level_data', 'fda_register']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.experimental_rerun()
                    
            if 'fuzzy_matched_data' in st.session_state and not st.session_state.fuzzy_matched_data.empty:
                st.write("Updated FDA Register with Fuzzy Matching and ATC Codes:")
                st.dataframe(st.session_state.fuzzy_matched_data)

                csv_data = convert_df_to_csv(st.session_state.fuzzy_matched_data)
                st.download_button(label="Download FDA Register as CSV", data=csv_data, file_name='fda_register_with_atc_codes.csv', mime='text/csv')
            else:
                st.write("No processed data available for download or processing not yet started.")
                
            if st.session_state.fuzzy_matched_data is not None:
                try:
                    st.session_state.fuzzy_matched_data = st.session_state.fuzzy_matched_data[['Ingredient', 'DF;Route', 'Strength', 'Trade_Name', 'Applicant', 'Best Match Name', 'Match Score', 'ATCCode']]

                    # Convert all strings in the DataFrame to uppercase
                    for column in st.session_state.fuzzy_matched_data.columns:
                        st.session_state.fuzzy_matched_data[column] = st.session_state.fuzzy_matched_data[column].map(lambda x: x.upper() if isinstance(x, str) else x)

                    # Assuming extract_atc_levels_human and extract_atc_levels_veterinary are defined
                    extract_atc_levels = extract_atc_levels_human if medicine_type == 'Human Medicine' else extract_atc_levels_veterinary

                    # Apply the function to each ATC code in the DataFrame
                    atc_data = st.session_state.fuzzy_matched_data['ATCCode'].apply(lambda x: pd.Series(extract_atc_levels(x)))
                    atc_data.columns = ['ATCLevelOneCode', 'ATCLevelTwoCode', 'ATCLevelThreeCode', 'ATCLevelFourCode']
                    st.session_state.fuzzy_matched_data = pd.concat([st.session_state.fuzzy_matched_data, atc_data], axis=1)

                    st.session_state.atc_level_data = st.session_state.fuzzy_matched_data

                    if not st.session_state.atc_level_data.empty:
                        st.write("Updated FDA Register with ATC Level Codes:")
                        st.dataframe(st.session_state.atc_level_data)

                        # Download file
                        csv = convert_df_to_csv(st.session_state.atc_level_data)
                        st.download_button(label="Download FDA Register as CSV", data=csv, file_name='fda_register_with_ATC_Level_Codes.csv', mime='text/csv', key='download_fda_updated_register')
                except KeyError as e:
                    print(f"Column not found in DataFrame: {e}")
            else:
                print("fda_register is None. Please check data loading and processing steps.")
                
            # Streamlit UI layout for ATC Code Description Integration and Filtering
            st.subheader("FDA Orange Book ATC Code Description Integration and Filtering")

            # Initialize session state for check_passed
            if 'check_passed' not in st.session_state:
                st.session_state['check_passed'] = False

            # Initialize variables for ATC data and filter variables
            atc_one = atc_two = atc_three = atc_four = None
            atc_one_desc = atc_two_desc = atc_three_desc = atc_four_desc = selected_generic_names = []

            # Required columns for each ATC level
            required_columns_atc_one = ['ATCLevelOneCode', 'ATCLevelOneDescript']
            required_columns_atc_two = ['ATCLevelTwoCode', 'ATCLevelTwoDescript']
            required_columns_atc_three = ['ATCLevelThreeCode', 'ATCLevelThreeDescript']
            required_columns_atc_four = ['ATCLevelFourCode', 'Chemical Subgroup']

            # File uploaders for ATC level description files
            atc_one_file = st.file_uploader("Upload ATC Level One Description File", type=['csv'], key="atc_one_uploader_one")
            atc_two_file = st.file_uploader("Upload ATC Level Two Description File", type=['csv'], key="atc_two_uploader_two")
            atc_three_file = st.file_uploader("Upload ATC Level Three Description File", type=['csv'], key="atc_three_uploader_three")
            atc_four_file = st.file_uploader("Upload ATC Level Four Description File", type=['csv'], key="atc_four_uploader_four")

            # Button to trigger the check operation
            check_data = st.button("Check Required Columns")

            if check_data:
                # Reset the check_passed flag
                st.session_state['check_passed'] = True

                # Load ATC description files if uploaded
                atc_one = safe_load_csv(atc_one_file) if atc_one_file else None
                atc_two = safe_load_csv(atc_two_file) if atc_two_file else None
                atc_three = safe_load_csv(atc_three_file) if atc_three_file else None
                atc_four = safe_load_csv(atc_four_file) if atc_four_file else None

                # Check for required columns in each ATC level description file
                check_required_columns(atc_one, required_columns_atc_one, "One")
                check_required_columns(atc_two, required_columns_atc_two, "Two")
                check_required_columns(atc_three, required_columns_atc_three, "Three")
                check_required_columns(atc_four, required_columns_atc_four, "Four")
                
            else:
                st.info("Please upload ATC level description files and press 'Check Required Columns'.")

            # Button to trigger the merge operation
            merge_data = st.button("Merge Data")

            if merge_data:
                if 'fuzzy_matched_data' in st.session_state and not st.session_state.fuzzy_matched_data.empty:
                    if st.session_state['check_passed']:  # Check if all required columns are present
                        # Your merging logic here...
                        # Load ATC description files if uploaded
                        atc_one = safe_load_csv(atc_one_file) if atc_one_file else None
                        atc_two = safe_load_csv(atc_two_file) if atc_two_file else None
                        atc_three = safe_load_csv(atc_three_file) if atc_three_file else None
                        atc_four = safe_load_csv(atc_four_file) if atc_four_file else None
                        # Merge with ATC level descriptions
                        with st.spinner('Merging data with ATC level descriptions...'):
                            merged_data = st.session_state.fuzzy_matched_data.copy()  # Work on a copy to prevent modifying the original data prematurely
                            if atc_one is not None and 'ATCLevelOneCode' in merged_data.columns:
                                merged_data = merged_data.merge(atc_one, on='ATCLevelOneCode', how='left')
                            if atc_two is not None and 'ATCLevelTwoCode' in merged_data.columns:
                                merged_data = merged_data.merge(atc_two, on='ATCLevelTwoCode', how='left')
                            if atc_three is not None and 'ATCLevelThreeCode' in merged_data.columns:
                                merged_data = merged_data.merge(atc_three, on='ATCLevelThreeCode', how='left')
                            if atc_four is not None and 'ATCLevelFourCode' in merged_data.columns:
                                merged_data = merged_data.merge(atc_four, on='ATCLevelFourCode', how='left')
                                
                        # Save the merged data in session state under a new key
                        st.session_state['fda_with_ATCCodeDescription'] = merged_data
                        # Display the merged dataframe
                        if not merged_data.empty:
                            st.write("Merged Data:")
                            st.dataframe(merged_data)
                        else:
                            st.write("No data to display after merging.")

                        st.success("Data merged with ATC level descriptions.")
                    else:
                        st.error("Cannot merge data. Please ensure all required columns are present and try again.")
                else:
                    st.warning("Please complete the fuzzy matching process and ensure ATC level description files are uploaded.")

            # Download file
            csv = convert_df_to_csv(st.session_state.fuzzy_matched_data)
            if csv is not None:
                # Proceed with operations that use 'csv'
                st.download_button(label="Download FDA Register as CSV", data=csv, file_name='fda_register_with_atc_description.csv', mime='text/csv', key='download_mcaz_withatcdescription_fda')

            else:
                # Handle the case where 'csv' is None, e.g., display a message or take alternative action
                print("No data available to convert to CSV")

            # Filter options presented to the user
            filter_options = ["None", "ATCLevelOneDescript", "ATCLevelTwoDescript", "ATCLevelThreeDescript", "Chemical Subgroup", "Ingredient"]
            selected_filter = st.radio("Select a filter", filter_options)

            # Check if 'fda_with_ATCCodeDescription' is in session state and is not empty
            if 'fda_with_ATCCodeDescription' in st.session_state and not st.session_state['fda_with_ATCCodeDescription'].empty:
                # Condition to handle filtering
                if selected_filter != "None":
                    # Convert all values in the selected filter column to string, get unique values, and sort
                    filter_values = sorted(st.session_state['fda_with_ATCCodeDescription'][selected_filter].astype(str).unique())
                    selected_values = st.multiselect(f"Select {selected_filter}", filter_values)

                    # Initialize filtered_data with the full dataset to handle scope
                    filtered_data = st.session_state['fda_with_ATCCodeDescription'].copy()

                    if selected_values:
                        # Apply filter based on selected values
                        filtered_data = filtered_data[filtered_data[selected_filter].astype(str).isin(selected_values)]
                        st.write(f"Filtered Data by {selected_filter}:")
                    else:
                        st.write("Displaying unfiltered data (no specific filter values selected):")

                    # Display filtered data and count
                    st.dataframe(filtered_data)
                    st.write(f"Filtered data count: {len(filtered_data)}")

                    # Offer CSV download for filtered data
                    csv = convert_df_to_csv(filtered_data)
                    st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='fda_register_filtered.csv', mime='text/csv', key='download_filtered')
                else:
                    st.write("No filter selected. Displaying unfiltered data:")
                    st.dataframe(st.session_state['fda_with_ATCCodeDescription'])
                    st.write(f"Total data count: {len(st.session_state['fda_with_ATCCodeDescription'])}")

                    # Optionally, offer download of unfiltered data
                    csv = convert_df_to_csv(st.session_state['fda_with_ATCCodeDescription'])
                    st.download_button(label="Download Unfiltered Data as CSV", data=csv, file_name='fda_register_unfiltered.csv', mime='text/csv', key='download_unfiltered')
            else:
                # Handle case where the data isn't available or hasn't been loaded
                st.write("Data not available. Please ensure data is loaded and processed.")
                
        # Applicant Analysis
        elif choice == 'Applicant Analysis':
            st.subheader('Applicant Analysis')
            
            # Anatomial Main Group
            st.subheader("Anatomcal Main Group Count")

            if 'fda_with_ATCCodeDescription' in st.session_state and not st.session_state['fda_with_ATCCodeDescription'].empty:
                fda_register = st.session_state['fda_with_ATCCodeDescription']

                # Remove complete duplicates
                fda_register = fda_register.drop_duplicates()

                if not fda_register.empty:
                    # Convert 'Principal Name' to string and handle NaN values
                    fda_register['Applicant'] = fda_register['Applicant'].fillna('Unknown').astype(str)

                    # Add "None" option and select Appplicant Name
                    applicant_options = ['None'] + sorted(fda_register['Applicant'].unique())
                    selected_applicant = st.selectbox("Select Applicant Name", applicant_options)

                    # Choose sort order
                    sort_order = st.radio("Select Sort Order", ["Ascending", "Descending"])

                    if selected_applicant != "None":
                        # Filter data based on selected principal
                        filtered_data = fda_register[fda_register['Applicant'] == selected_applicant]
                    else:
                        # If "None" is selected, use the entire dataset
                        filtered_data = fda_register

                    # Group by 'ATC Level One Description' and count unique 'Generic Name'
                    # Sort based on the selected sort order
                    if sort_order == "Descending":
                        atc_classification_count = (filtered_data.groupby('ATCLevelOneDescript')['Ingredient']
                                                    .count()
                                                    .reset_index()
                                                    .rename(columns={'Ingredient': 'IngredientCount'})
                                                    .sort_values(by='IngredientCount', ascending=False))
                    else:
                        atc_classification_count = (filtered_data.groupby('ATCLevelOneDescript')['Ingredient']
                                                    .count()
                                                    .reset_index()
                                                    .rename(columns={'Ingredient': 'IngredientCount'})
                                                    .sort_values(by='IngredientCount', ascending=True))

                    st.write(f"Count of Ingredient Name by ATC Level One Description (sorted {sort_order}):")
                    st.dataframe(atc_classification_count)

                    # Calculate the total count of products across all ATC Level One Descriptions
                    # Ensure you're summing only the 'GenericNameCount' column
                    total_product_count = atc_classification_count['IngredientCount'].sum()
                    st.write(f"Total Count of Products (Across All Groups): {total_product_count}")

                    # Convert the complete DataFrame to CSV
                    csv_data = convert_df_to_csv(atc_classification_count)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv_data,
                        file_name='atc_classification_one_count_orange.csv',
                        mime='text/csv',
                    )

                else:
                    st.write("No data available.")
            else:
                st.write("ATC Code Description data is not available.")

            # Pharmacological Group
            st.subheader("Pharmacological Group Count")

            if 'fda_with_ATCCodeDescription' in st.session_state and not st.session_state['fda_with_ATCCodeDescription'].empty:
                fda_register = st.session_state['fda_with_ATCCodeDescription']

                # Remove complete duplicates
                fda_register = fda_register.drop_duplicates()

                if not fda_register.empty:
                    # Convert 'Principal Name' to string and handle NaN values
                    fda_register['Applicant'] = fda_register['Applicant'].fillna('Unknown').astype(str)

                    # Add "None" option and select Principal Name
                    applicant_options = ['None'] + sorted(fda_register['Applicant'].unique())
                    selected_applicant_3 = st.selectbox("Select Applicant Name", applicant_options, key = "applicant_selection_3")
                    
                    # Choose sort order
                    sort_order_3 = st.radio("Select Sort Order", ["Ascending", "Descending"], key = "sort_order_selection_3")

                    if selected_applicant_3 != "None":
                        # Filter data based on selected principal
                        filtered_data = fda_register[fda_register['Applicant'] == selected_applicant_3]
                    else:
                        # If "None" is selected, use the entire dataset
                        filtered_data = fda_register

                    # Group by 'ATC Level One Description' and count unique 'Generic Name'
                    # Sort based on the selected sort order
                    if sort_order_3 == "Descending":
                        atc_classification_count = (filtered_data.groupby('ATCLevelTwoDescript')['Ingredient']
                                                    .count()
                                                    .reset_index()
                                                    .rename(columns={'Ingredient': 'IngredientCount'})
                                                    .sort_values(by='IngredientCount', ascending=False))
                    else:
                        atc_classification_count = (filtered_data.groupby('ATCLevelTwoDescript')['Ingredient']
                                                    .count()
                                                    .reset_index()
                                                    .rename(columns={'Ingredient': 'IngredientCount'})
                                                    .sort_values(by='IngredientCount', ascending=True))

                    st.write(f"Count of Ingredient Name by ATC Level Two Description (sorted {sort_order_3}):")
                    st.dataframe(atc_classification_count)

                    # Calculate the total count of products across all ATC Level One Descriptions
                    # Ensure you're summing only the 'GenericNameCount' column
                    total_product_count = atc_classification_count['IngredientCount'].sum()
                    st.write(f"Total Count of Products (Across All Groups): {total_product_count}")

                    # Convert the complete DataFrame to CSV
                    csv_data = convert_df_to_csv(atc_classification_count)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv_data,
                        file_name='atc_classification_two_count_orange.csv',
                        mime='text/csv', key = "pharmacology",
                    )

                else:
                    st.write("No data available.")
            else:
                st.write("ATC Code Description data is not available.")

            # Therapuetic Group
            st.subheader("Therapeutic Group Count")

            if 'fda_with_ATCCodeDescription' in st.session_state and not st.session_state['fda_with_ATCCodeDescription'].empty:
                fda_register = st.session_state['fda_with_ATCCodeDescription']

                # Remove complete duplicates
                fda_register = fda_register.drop_duplicates()

                if not fda_register.empty:
                    # Convert 'Applicant Name' to string and handle NaN values
                    fda_register['Applicant'] = fda_register['Applicant'].fillna('Unknown').astype(str)

                    # Add "None" option and select Applicant Name
                    applicant_options = ['None'] + sorted(fda_register['Applicant'].unique())
                    selected_applicant_4 = st.selectbox("Select Applicant Name", applicant_options, key = "applicant_applicant_4")

                    # Choose sort order
                    sort_order_4 = st.radio("Select Sort Order", ["Ascending", "Descending"], key = "sort_order_selection_4")

                    if selected_applicant_4 != "None":
                        # Filter data based on selected applicant
                        filtered_data = fda_register[fda_register['Applicant'] == selected_applicant_4]
                    else:
                        # If "None" is selected, use the entire dataset
                        filtered_data = fda_register

                    # Group by 'ATC Level One Description' and count unique 'Generic Name'
                    # Sort based on the selected sort order
                    if sort_order_4 == "Descending":
                        atc_classification_count = (filtered_data.groupby('ATCLevelThreeDescript')['Ingredient']
                                                    .count()
                                                    .reset_index()
                                                    .rename(columns={'Ingredient': 'IngredientCount'})
                                                    .sort_values(by='IngredientCount', ascending=False))
                    else:
                        atc_classification_count = (filtered_data.groupby('ATCLevelThreeDescript')['Ingredient']
                                                    .count()
                                                    .reset_index()
                                                    .rename(columns={'Ingredient': 'IngredientCount'})
                                                    .sort_values(by='IngredientCount', ascending=True))

                    st.write(f"Count of Ingredient Name by ATC Level Three Description (sorted {sort_order_4}):")
                    st.dataframe(atc_classification_count)

                    # Calculate the total count of products across all ATC Level One Descriptions
                    # Ensure you're summing only the 'GenericNameCount' column
                    total_product_count = atc_classification_count['IngredientCount'].sum()
                    st.write(f"Total Count of Products (Across All Groups): {total_product_count}")

                    # Convert the complete DataFrame to CSV
                    csv_data = convert_df_to_csv(atc_classification_count)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv_data,
                        file_name='atc_classification_three_count_orange.csv',
                        mime='text/csv', key = "therapeutic",
                    )

                else:
                    st.write("No data available.")
            else:
                st.write("ATC Code Description data is not available.")
               
        
        # Patient Flow Forecasting
        elif choice == 'Patient-flow Forecast':
            st.subheader('Patient-flow Forecast')
            # Implement Patient flow Forecast

            # Input fields
            population = st.number_input("Population (millions)", min_value=0.0, value=1.0, step=0.1)
            prevalence = st.number_input("Epidemiology (prevalence %)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
            symptomatic_rate = st.number_input("Symptomatic rate (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
            diagnosis_rate = st.number_input("Diagnosis rate (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
            access_rate = st.number_input("Access rate (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
            treatment_rate = st.number_input("Drug-treated patients (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)

            if st.button("Calculate"):
                prevalent_population = calculate_prevalent_population(population, prevalence)
                symptomatic_population = calculate_symptomatic_population(prevalent_population, symptomatic_rate)
                diagnosed_population = calculate_diagnosed_population(symptomatic_population, diagnosis_rate)
                potential_patients = calculate_potential_patients(diagnosed_population, access_rate)
                drug_treated_patients = calculate_drug_treated_patients(potential_patients, treatment_rate)

                st.write(f"Prevalent Population: {prevalent_population} million")
                st.write(f"Symptomatic Population: {symptomatic_population} million")
                st.write(f"Diagnosed Population: {diagnosed_population} million")
                st.write(f"Potential Patients: {potential_patients} million")
                st.write(f"Drug-treated Patients: {drug_treated_patients} million")

        # Drug Classification Analysis
        elif choice == 'Drug Classification Analysis':
            st.subheader('Drug Classification Analysis')
            # Implement drug classification analysis

            # Assume mcaz_register is loaded elsewhere in your application
            # Load mcaz_register
            mcaz_register = load_data(uploaded_file)

            # Filter options for 'Categories of Distribution'
            categories_options = ['All Categories'] + sorted(mcaz_register['Categories for Distribution'].dropna().unique().tolist())
            selected_category = st.selectbox('Select Category for Distribution', categories_options, index=0)

            # Filter options for 'Manufacturers'
            manufacturers_options = ['All Manufacturers'] + sorted(mcaz_register['Manufacturers'].dropna().unique().tolist())
            selected_manufacturer = st.selectbox('Select Manufacturer', manufacturers_options, index=0)

            # Apply filters
            if selected_category != 'All Categories':
                mcaz_register = mcaz_register[mcaz_register['Categories for Distribution'] == selected_category]

            if selected_manufacturer != 'All Manufacturers':
                mcaz_register = mcaz_register[mcaz_register['Manufacturers'] == selected_manufacturer]

            # Display filtered data
            st.write("Filtered Data:")
            st.dataframe(mcaz_register)

            # Display the total count of products
            total_products = len(mcaz_register)
            st.write(f"Total Count of Products: {total_products}")
            
            # Summary of Categories for Distribution
            def summarize_categories_by_principal(mcaz_register):
                st.subheader("Summary of Categories for Distribution by Principal")

                # Check for necessary columns
                required_columns = ['Principal Name', 'Categories for Distribution', 'Date Registered']
                if not all(col in mcaz_register.columns for col in required_columns):
                    st.error("Uploaded data does not contain the required columns.")
                    return

                # Preprocess data
                data = mcaz_register.dropna(subset=required_columns)
                data['Year'] = pd.to_datetime(data['Date Registered'], errors='coerce').dt.year

                # Principal selection
                principal_names = ['All'] + sorted(data['Principal Name'].unique())
                selected_principal = st.selectbox('Select Principal Name', principal_names)

                # Filter data
                if selected_principal != 'All':
                    filtered_data = data[data['Principal Name'] == selected_principal]
                else:
                    filtered_data = data

                # Initial summary
                total_product_count = filtered_data.shape[0]
                category_counts = filtered_data['Categories for Distribution'].value_counts().reset_index(name='Count')
                category_counts['% Total'] = (category_counts['Count'] / total_product_count) * 100
                category_counts.columns = ['Category', 'Count', '% Total']

                # Display initial summary
                if not category_counts.empty:
                    st.write(f"Initial Summary for {selected_principal}:")
                    st.dataframe(category_counts.style.format({'% Total': "{:.2f}%"}))
                    st.markdown(f"**Total Product Count:** {total_product_count}")

                # Detailed yearly summary
                st.write(f"Yearly Summary for {selected_principal}:")

                # Group and calculate counts and percentages
                grouped = filtered_data.groupby(['Categories for Distribution', 'Year']).size().reset_index(name='Count')
                total_counts_by_year = filtered_data.groupby(['Year', 'Categories for Distribution']).size().groupby(level=0).sum().reset_index(name='TotalYearCount')

                # Merge for percentages
                summary = pd.merge(grouped, total_counts_by_year, on='Year')
                summary['% Total'] = (summary['Count'] / summary['TotalYearCount']) * 100

                # Pivot for yearly summary with specific formatting
                pivot_df = summary.pivot_table(index='Categories for Distribution', columns='Year', values=['Count', '% Total'], aggfunc='first')

                # Create multi-level columns for each year with "Total Count" and "% For the Year"
                pivot_df.columns = pivot_df.columns.map('{0[0]} {0[1]}'.format)
                pivot_df = pivot_df.sort_index(axis=1, level=1)

                # Rearrange columns to have "Count" and "% Total" next to each other for each year
                new_order = []
                years = np.unique([col.split(' ')[-1] for col in pivot_df.columns])
                for year in sorted(years, key=int):
                    new_order.extend(sorted([col for col in pivot_df.columns if year in col], key=lambda x: x.split()[0], reverse=True))
                pivot_df = pivot_df[new_order]

                # Format percentages to two decimal places
                for col in pivot_df.columns:
                    if "% Total" in col:
                        pivot_df[col] = pivot_df[col].map("{:.2f}%".format)

                # Display the formatted pivot table
                if not pivot_df.empty:
                    st.dataframe(pivot_df)
                else:
                    st.write("No yearly data available.")
        
            summarize_categories_by_principal(mcaz_register)
        
        # Drugs with No Patents and NO Competition Analysis
        elif choice == 'Drugs with no Competition':
            st.subheader('FDA Drugs with No Patents and No Competition')
            # Implement FDA No Patents analysis

            # Medicine type selection
            medicine_type = st.radio("Select Medicine Type", ["Human Medicine", "Veterinary Medicine"])
                                    
            # Load MCAZ Register data from session state or initialize if not present
            mcaz_register = st.session_state.get('mcaz_register', pd.DataFrame())

            if medicine_type == "Human Medicine":
                uploaded_file = st.file_uploader("Upload your Drugs with No Patents No Competition file", type=['csv'])

                if uploaded_file is not None:
                    # Load data into session state
                    st.session_state['fda_data'] = load_data_fda(uploaded_file)
                    fda_data = st.session_state['fda_data']

                    if not fda_data.empty and not mcaz_register.empty:
                        # Filter out products that are in the MCAZ Register
                        filtered_fda_data = filter_fda_data(fda_data, mcaz_register)
                        st.session_state['filtered_fda_data'] = filtered_fda_data  # Store filtered data in session state

                        # Add "None" option and sort filter options
                        dosage_form_options = ['None'] + sorted(filtered_fda_data['DOSAGE FORM'].dropna().unique().tolist())
                        selected_dosage_form = st.selectbox("Select Dosage Form", dosage_form_options)

                        type_options = ['None'] + sorted(filtered_fda_data['TYPE'].dropna().unique().tolist())
                        selected_type = st.selectbox("Select Type", type_options)

                        # Apply filters if selections are not "None"
                        if selected_dosage_form != "None":
                            st.session_state['filtered_fda_data'] = st.session_state['filtered_fda_data'][st.session_state['filtered_fda_data']['DOSAGE FORM'] == selected_dosage_form]
                        if selected_type != "None":
                            st.session_state['filtered_fda_data'] = st.session_state['filtered_fda_data'][st.session_state['filtered_fda_data']['TYPE'] == selected_type]

                        # Display the filtered dataframe
                        st.write("Filtered FDA Data (Excluding MCAZ Registered Products):")
                        st.dataframe(st.session_state['filtered_fda_data'])

                        # Count and display the number of drugs
                        drug_count = len(st.session_state['filtered_fda_data'])
                        st.write(f"Total Number of Unique Drugs: {drug_count}")

                        # Convert the complete DataFrame to CSV
                        csv_data = convert_df_to_csv(st.session_state['filtered_fda_data'])
                        st.download_button(
                            label="Download data as CSV",
                            data=csv_data,
                            file_name='fda_nocompetition_product_count.csv',
                            mime='text/csv',
                        )
                    else:
                        st.write("Upload a file to see the data or ensure MCAZ Register data is available.")
                else:
                    st.write("Please upload a file.")
            else:
                st.write("Select 'Human Medicine' to access FDA drugs analysis.")

        # Top Phamra Companies Word wide Sales
        elif choice == 'Top Pharma Companies Sales':
            st.subheader('Top Pharma Companies World Sales')
            
            # Initialize session state for DataFrame if not already present
            if 'df' not in st.session_state:
                st.session_state.df = pd.DataFrame()

            # Upload functionality
            uploaded_file = st.file_uploader("Upload your sales data CSV file", type=["csv"])
            if uploaded_file is not None:
                # Load data into session state only if a new file is uploaded
                st.session_state.df = load_data_sales(uploaded_file)

            # Check if the DataFrame is not empty
            if not st.session_state.df.empty:
                st.subheader('Filter Options')

                # Dynamic lists for filter options
                company_list = ['All'] + sorted(st.session_state.df['Company Name'].unique().tolist())
                product_list = ['All'] + sorted(st.session_state.df['Product Name'].unique().tolist())
                ingredient_list = ['All'] + sorted(st.session_state.df['Active Ingredient'].fillna('Unknown').unique().tolist())
                indication_list = ['All'] + sorted(st.session_state.df['Main Therapeutic Indication'].fillna('Unknown').unique().tolist())
                classification_list = ['All'] + sorted(st.session_state.df['Product Classification'].fillna('Unknown').unique().tolist())

                # User filter selections
                company_name = st.selectbox('Company Name', company_list)
                product_name = st.selectbox('Product Name', product_list)
                active_ingredient = st.selectbox('Active Ingredient', ingredient_list)
                therapeutic_indication = st.selectbox('Main Therapeutic Indication', indication_list)
                product_classification = st.selectbox('Product Classification', classification_list)

                # Sorting options
                sort_column = st.selectbox('Sort by', ['2022 Revenue in Millions USD', '2021 Revenue in Millions USD'], index=0)
                sort_order = st.selectbox('Sort order', ['Ascending', 'Descending'], index=1)
                is_ascending = sort_order == 'Ascending'

                # Apply filters to a local copy of the DataFrame
                filtered_df = st.session_state.df.copy()
                if company_name != 'All':
                    filtered_df = filtered_df[filtered_df['Company Name'] == company_name]
                if product_name != 'All':
                    filtered_df = filtered_df[filtered_df['Product Name'] == product_name]
                if active_ingredient != 'All':
                    filtered_df = filtered_df[filtered_df['Active Ingredient'] == active_ingredient]
                if therapeutic_indication != 'All':
                    filtered_df = filtered_df[filtered_df['Main Therapeutic Indication'] == therapeutic_indication]
                if product_classification != 'All':
                    filtered_df = filtered_df[filtered_df['Product Classification'] == product_classification]

                # Sort and display the filtered DataFrame
                if not filtered_df.empty:
                    filtered_df = filtered_df.sort_values(by=sort_column, ascending=is_ascending)
                    st.write(filtered_df)
                else:
                    st.write("No data to display after filtering.")

                # Download button for filtered data
                if not filtered_df.empty:
                    csv_data = convert_df_to_csv(filtered_df)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv_data,
                        file_name='filtered_data.csv',
                        mime='text/csv',
                    )
            else:
                st.write("Please upload a sales data CSV file to get started.")
        
        # FDA Drug Establishment Sites
        elif choice == 'FDA Drug Establishment Sites':
            st.subheader('FDA Drug Establishment Sites')
            
            # File uploader for the Establishment file
            establishment_file = st.file_uploader("Choose an Establishment CSV file", type="csv")
            # File uploader for the Country Codes file
            country_codes_file = st.file_uploader("Choose a Country Codes CSV file", type="csv")

            if establishment_file is not None and country_codes_file is not None:
                # Process the uploaded files
                df = process_uploaded_file(establishment_file)
                country_codes_df = pd.read_csv(country_codes_file)

                # Merge the establishment dataframe with the country codes dataframe
                merged_df = df.merge(country_codes_df, left_on='COUNTRY_CODE', right_on='Alpha-3 code', how='left')

                # Ensure all values are strings for sorting and filtering
                merged_df.fillna('Unknown', inplace=True)

                # Dropdowns for filtering with sorted options
                firm_name_options = sorted(merged_df['FIRM_NAME'].unique().tolist())
                country_options = sorted(merged_df['Country'].unique().tolist())  # Changed to 'Country'
                operations_options = sorted(merged_df['OPERATIONS'].unique().tolist(), key=lambda x: (x is np.nan, x))
                registrant_name_options = sorted(merged_df['REGISTRANT_NAME'].unique().tolist())

                firm_name = st.selectbox("Firm Name", ["All"] + firm_name_options)
                country = st.selectbox("Country", ["All"] + country_options)  # Changed to 'Country'
                operations = st.selectbox("Operations", ["All"] + operations_options)
                registrant_name = st.selectbox("Registrant Name", ["All"] + registrant_name_options)

                # Filter the dataframe based on selection
                filtered_df = filter_dataframe_establishments(merged_df, firm_name, country, operations, registrant_name)

                # Save the filtered dataframe in the session state for persistence across modules
                st.session_state.filtered_data = filtered_df

                st.dataframe(filtered_df)

                # Download button for the filtered dataframe
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download filtered data as CSV",
                    data=csv,
                    file_name='filtered_fda_sites.csv',
                    mime='text/csv',
                )
                
       
        # FDA NME & New Biologic Approvals
        if choice == 'FDA NME & New Biologic Approvals':
            st.subheader('FDA NME & New Biologic Approvals')

            uploaded_file = st.file_uploader("Choose an NME & New Biologics file")
            if uploaded_file is not None:
                # Process and store the uploaded data only if a new file is provided
                df_filtered = load_data_nme(uploaded_file)
                # Reset filters if a new file is uploaded
                st.session_state['nme_biologics_data'] = df_filtered
                st.session_state['nme_biologics_filters'] = {}
            elif 'nme_biologics_data' in st.session_state:
                # Use previously loaded data
                df_filtered = st.session_state['nme_biologics_data']
            else:
                st.warning("Please upload a file to begin.")
                st.stop()

            # Initialize or retrieve filter settings from session state
            filter_settings = st.session_state.get('nme_biologics_filters', {})

            # Define UI for all filters and update filter_settings based on user input
            # Approval Year Range
            if 'Approval Year' in df_filtered:
                year_options = range(int(df_filtered['Approval Year'].min()), int(df_filtered['Approval Year'].max()) + 1)
                start_year, end_year = st.select_slider(
                    'Select Approval Year Range:',
                    options=list(year_options),
                    value=filter_settings.get('year_range', (min(year_options), max(year_options)))
                )
                filter_settings['year_range'] = (start_year, end_year)

            # NDA/BLA
            nda_bla_options = ['All'] + sorted(df_filtered['NDA/BLA'].unique().tolist())
            nda_bla_selection = st.selectbox('NDA/BLA', options=nda_bla_options, index=0)
            filter_settings['nda_bla_selection'] = nda_bla_selection

            # Active Ingredient/Moiety
            active_ingredient_options = ['All'] + sorted(df_filtered['Active Ingredient/Moiety'].unique().tolist())
            active_ingredient_selection = st.selectbox('Active Ingredient/Moiety', options=active_ingredient_options, index=0)
            filter_settings['active_ingredient_selection'] = active_ingredient_selection

            # Additional Filters
            review_designation_options = ['All', 'Priority', 'Standard']
            review_designation_selection = st.selectbox('Review Designation', options=review_designation_options, index=0)
            filter_settings['review_designation_selection'] = review_designation_selection

            orphan_drug_option = st.checkbox('Orphan Drug Designation', value='Orphan Drug Designation' in filter_settings)
            filter_settings['orphan_drug_option'] = orphan_drug_option

            accelerated_approval_option = st.checkbox('Accelerated Approval', value='Accelerated Approval' in filter_settings)
            filter_settings['accelerated_approval_option'] = accelerated_approval_option

            breakthrough_therapy_option = st.checkbox('Breakthrough Therapy Designation', value='Breakthrough Therapy Designation' in filter_settings)
            filter_settings['breakthrough_therapy_option'] = breakthrough_therapy_option

            fast_track_option = st.checkbox('Fast Track Designation', value='Fast Track Designation' in filter_settings)
            filter_settings['fast_track_option'] = fast_track_option

            qualified_infectious_option = st.checkbox('Qualified Infectious Disease Product', value='Qualified Infectious Disease Product' in filter_settings)
            filter_settings['qualified_infectious_option'] = qualified_infectious_option

            # Apply filters based on user selection
            df_filtered = apply_all_filters(df_filtered, filter_settings)

            # Update session state with the latest filter settings
            st.session_state['nme_biologics_filters'] = filter_settings

            # Display the filtered dataframe
            st.dataframe(df_filtered)
            st.write(f"Filtered data count: {len(df_filtered)}")

            # Download button for the filtered dataframe
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name='filtered_fda_nmes_biologics.csv',
                mime='text/csv',
            )
    
                
        # Assuming 'choice' variable is determined by some user interaction upstream in your code
        if choice == 'EMA FDA Health Canada Approvals 2023':
            st.subheader('EMA FDA Health Canada Approvals 2023')
                        
            uploaded_file = st.file_uploader("Choose a EMA FDA Health Canada 2023 Approvals CSV file", type="csv")

            if uploaded_file is not None:
                try:
                    # Directly read the uploaded file
                    data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                    # Store the processed data in session_state
                    st.session_state[data_key] = data
                except Exception as e:
                    st.error(f'Failed to process the uploaded file: {e}')
                    return
            elif data_key in st.session_state:
                # Use the previously uploaded and processed data
                data = st.session_state[data_key]
            else:
                st.warning("Please upload a file to proceed.")
                return
            
            # Provide an option to re-upload and clear the existing data
            if st.button('Clear data'):
                if data_key in st.session_state:
                    del st.session_state[data_key]
                st.experimental_rerun()

            # Initialize or retrieve filter states from session state
            filter_defaults = {
                'drug_name': 'All', 'company_name': 'All', 'active_ingredient': 'All', 'therapeutic_area': 'All',
                'product_type': 'All', 'regulatory_authority': 'All', 'application_type': 'All', 'drug_type': 'All'
            }
            for filter_key, default_value in filter_defaults.items():
                if filter_key not in st.session_state:
                    st.session_state[filter_key] = default_value

            # Create filter selectors
            st.session_state['drug_name'] = st.selectbox('Drug Name', ['All'] + sorted(data['Drug Name'].unique().tolist()), index=0)
            st.session_state['company_name'] = st.selectbox('Company Name', ['All'] + sorted(data['Company Name'].unique().tolist()), index=0)
            st.session_state['active_ingredient'] = st.selectbox('Active Ingredient', ['All'] + sorted(data['Active Ingredient'].unique().tolist()), index=0)
            st.session_state['therapeutic_area'] = st.selectbox('Therapeutic Area', ['All'] + sorted(data['Therapeutic Area'].unique().tolist()), index=0)
            st.session_state['product_type'] = st.selectbox('Product Type', ['All'] + sorted(data['Product Type'].unique().tolist()), index=0)
            st.session_state['regulatory_authority'] = st.selectbox('Regulatory Authority', ['All'] + sorted(data['Regulatory Authority'].unique().tolist()), index=0)
            st.session_state['application_type'] = st.selectbox('Application Type', ['All'] + sorted(data['Application Type'].unique().tolist()), index=0)
            st.session_state['drug_type'] = st.selectbox('Drug Type', ['All'] + sorted(data['Drug Type'].unique().tolist()), index=0)

            # Apply filters
            filtered_data = data
            if st.session_state['drug_name'] != 'All':
                filtered_data = filtered_data[filtered_data['Drug Name'] == st.session_state['drug_name']]
            if st.session_state['company_name'] != 'All':
                filtered_data = filtered_data[filtered_data['Company Name'] == st.session_state['company_name']]
            if st.session_state['active_ingredient'] != 'All':
                filtered_data = filtered_data[filtered_data['Active Ingredient'] == st.session_state['active_ingredient']]
            if st.session_state['therapeutic_area'] != 'All':
                filtered_data = filtered_data[filtered_data['Therapeutic Area'] == st.session_state['therapeutic_area']]
            if st.session_state['product_type'] != 'All':
                filtered_data = filtered_data[filtered_data['Product Type'] == st.session_state['product_type']]
            if st.session_state['regulatory_authority'] != 'All':
                filtered_data = filtered_data[filtered_data['Regulatory Authority'] == st.session_state['regulatory_authority']]
            if st.session_state['application_type'] != 'All':
                filtered_data = filtered_data[filtered_data['Application Type'] == st.session_state['application_type']]
            if st.session_state['drug_type'] != 'All':
                filtered_data = filtered_data[filtered_data['Drug Type'] == st.session_state['drug_type']]

            # Drop specified columns, if they exist, and display the filtered dataset
            columns_to_remove = ['Product Status Link', 'Estimated Sales (mm USD) Link']
            filtered_data = filtered_data.drop(columns=columns_to_remove, errors='ignore')

            st.write(filtered_data)

            # Display the count of the filtered dataframe
            st.write(f'Count of filtered results: {len(filtered_data)}')

            # Button to save the filtered data as CSV
            csv = filtered_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download filtered data as CSV",
                data=csv,
                file_name='filtered_fda_ema_healthcanada.csv',
                mime='text/csv',
            )
       
        # Assuming 'choice' variable is determined by some user interaction upstream in your code
        if choice == 'Drugs@FDA Analysis':
            
            # Simplified session state management
            if choice:
                st.session_state['selected_analysis'] = choice

            # Conditional execution based on session state
            if 'selected_analysis' in st.session_state:
                if st.session_state['selected_analysis'] == "Drugs@FDA Analysis":
                    perform_drugs_fda_analysis()
              
    else:
        st.warning('Please upload MCAZ Register CSV file.')

def main():
    # Password input
    st.image("logo.png", width=200)
    st.markdown("<h1 style='font-size:30px;'>Pharmaceutical Products Analysis Application</h1>", unsafe_allow_html=True)
    password_guess = st.text_input('What is the Password?', type="password").strip()

    # Check if password is entered and incorrect
    if password_guess and password_guess != st.secrets["password"]:
        st.error("Incorrect password. Please try again.")
        st.stop()

    # Check if password is correct
    if password_guess == st.secrets["password"]:
        try:
            # Correctly using datetime.strptime now
            # expiration_date = st.secrets["expiration_date"]
            expiration_date = datetime.strptime(st.secrets["expiration_date"], "%d-%m-%Y")

        except Exception as e:
            st.error(f"Error parsing expiration date: {e}")
            st.stop()
            return

        if datetime.now() > expiration_date:
            st.error("Product license has expired. Please contact the administrator.")
            st.stop()
        else:
            st.success("Password is correct and license has not expired")

        # Display main application content if the user is logged in and the password is not expired
        display_main_application_content()

if __name__ == "__main__":
    main()



# In[ ]:




