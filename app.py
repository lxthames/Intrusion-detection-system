# main.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Automated Digital Forensics Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Loading and Caching ---
# Column names for the NSL-KDD dataset
# Found from dataset documentation
KDD_COL_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

@st.cache_data(ttl=3600)
def load_data():
    """
    Loads, caches, and performs initial processing of the NSL-KDD dataset.
    """
    try:
        # Load training and testing data from URLs
        train_url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt'
        test_url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'
        
        train_df = pd.read_csv(train_url, header=None, names=KDD_COL_NAMES)
        test_df = pd.read_csv(test_url, header=None, names=KDD_COL_NAMES)

        # Drop the 'difficulty' column as it's not needed for classification
        train_df.drop('difficulty', axis=1, inplace=True)
        test_df.drop('difficulty', axis=1, inplace=True)

        # Create a unified 'attack_type' column for easier analysis
        # If label is 'normal', it's 'normal', otherwise it's an 'attack'
        train_df['attack_type'] = train_df['label'].apply(lambda r: 'normal' if r == 'normal' else 'attack')
        test_df['attack_type'] = test_df['label'].apply(lambda r: 'normal' if r == 'normal' else 'attack')

        return train_df, test_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Could not fetch dataset. Please check the network connection and the data source URL.")
        return None, None

@st.cache_resource
def train_model(_model_name, X_train_processed, y_train):
    """
    Initializes, trains, and caches a specified classifier model.
    Using _model_name as the first argument helps st.cache_resource to cache based on its value.
    """
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Support Vector Machine": SVC(random_state=42)
    }
    model = models[_model_name]
    
    # SVM and Logistic Regression benefit from feature scaling
    if _model_name in ["Logistic Regression", "Support Vector Machine"]:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train_processed)

    model.fit(X_train_processed, y_train)
    return model

def preprocess_data(df_train, df_test):
    """
    Preprocesses the data using Label Encoding for categorical features.
    """
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Identify categorical columns for encoding
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('label') # Keep original labels for reference
    categorical_cols.remove('attack_type')

    # Apply Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Separate back into train and test sets
    train_processed = df.iloc[:len(df_train)]
    test_processed = df.iloc[len(df_train):]

    # Prepare features (X) and target (y)
    X_train = train_processed.drop(['label', 'attack_type'], axis=1)
    y_train = train_processed['attack_type']
    X_test = test_processed.drop(['label', 'attack_type'], axis=1)
    y_test = test_processed['attack_type']

    return X_train, y_train, X_test, y_test, label_encoders

# --- Main Application ---
def main():
    """
    The main function that runs the Streamlit application.
    """
    st.title("🛡️ Automated Digital Forensics Framework")
    st.markdown("""
    This application is a prototype for the dissertation *'Development of an Automated Digital Forensics Framework for Efficient Analysis of Network Intrusions'*. 
    It uses the **NSL-KDD dataset** to demonstrate key functionalities like log analysis and machine learning-based intrusion detection.
    """)

    # Load data
    train_df, test_df = load_data()

    if train_df is None or test_df is None:
        st.stop()

    # --- Sidebar Navigation ---
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", [
        "🏠 Dashboard",
        "📊 Log and Data Analysis",
        "🤖 ML Intrusion Detection",
        "📄 Generate Forensic Report"
    ])

    # Preprocess data once and use it across pages
    X_train, y_train, X_test, y_test, encoders = preprocess_data(train_df, test_df)

    # --- Page 1: Dashboard ---
    if page == "🏠 Dashboard":
        st.header("Project Dashboard")
        st.markdown("### Overview")
        st.write("This dashboard provides a high-level summary of the network intrusion dataset.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Training Records", f"{len(train_df):,}")
        col2.metric("Testing Records", f"{len(test_df):,}")
        col3.metric("Total Features", len(train_df.columns) - 2) # Exclude label and attack_type

        st.markdown("### Attack Distribution in Training Data")
        attack_counts = train_df['label'].value_counts()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=attack_counts.index, y=attack_counts.values, ax=ax, palette="viridis")
        ax.set_title("Distribution of Connection Types (Normal vs. Specific Attacks)")
        ax.set_ylabel("Number of Records")
        ax.set_xlabel("Connection Label")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        
        st.info("The NSL-KDD dataset contains various types of network intrusions, categorized for analysis. The goal of our model is to distinguish 'normal' traffic from any form of 'attack'.")


    # --- Page 2: Log and Data Analysis ---
    elif page == "📊 Log and Data Analysis":
        st.header("Log and Data Analysis")
        st.markdown("Explore the raw network data, simulating the process of a forensic analyst examining logs.")

        if st.checkbox("Show Raw Training Data", value=False):
            st.dataframe(train_df)
        
        st.markdown("### Filter Data by Connection Type")
        selected_label = st.selectbox("Select a connection label to inspect:", ['All'] + sorted(train_df['label'].unique().tolist()))

        if selected_label == 'All':
            st.dataframe(train_df.head(1000))
        else:
            filtered_data = train_df[train_df['label'] == selected_label]
            st.write(f"Displaying records for **{selected_label}** connections:")
            st.dataframe(filtered_data)

    # --- Page 3: ML Intrusion Detection ---
    elif page == "🤖 ML Intrusion Detection":
        st.header("Machine Learning-Based Intrusion Detection")
        st.markdown("This section demonstrates automated analysis using a choice of classifier models.")

        # Model selection in the sidebar for this page
        st.sidebar.header("Model Selection")
        model_name = st.sidebar.selectbox(
            "Choose a model",
            ("Random Forest", "Logistic Regression", "Decision Tree", "Support Vector Machine"),
            index=0 # Default to Random Forest
        )
        
        st.info(f"You have selected the **{model_name}** model.")

        with st.spinner(f"Training the {model_name} model... This may take a moment."):
            model = train_model(model_name, X_train, y_train)
        
        st.success(f"{model_name} model training complete!")

        # Scale test data if necessary
        X_test_scaled = X_test.copy()
        if model_name in ["Logistic Regression", "Support Vector Machine"]:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train) # Fit on train
            X_test_scaled = scaler.transform(X_test) # Transform test

        # Make predictions on the test set
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric(f"Accuracy ({model_name})", f"{accuracy:.2%}")
        
        st.markdown("#### Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.markdown("#### Confusion Matrix")
        st.write(f"The confusion matrix shows the {model_name} model's performance in distinguishing between normal and attack connections.")
        cm = confusion_matrix(y_test, y_pred, labels=['normal', 'attack'])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

        # Feature importance is only available for tree-based models
        if model_name in ["Random Forest", "Decision Tree"]:
            st.subheader("Feature Importance")
            st.write("The chart below shows the most influential factors the model uses to detect an intrusion.")
            feature_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax, palette='mako')
            ax.set_title("Top 15 Most Important Features")
            st.pyplot(fig)


    # --- Page 4: Generate Forensic Report ---
    elif page == "📄 Generate Forensic Report":
        st.header("Generate Forensic Report")
        st.markdown("This section compiles the findings into a summary report for documentation, using the default **Random Forest** model for consistency.")
        
        # We need the model's predictions to generate the report
        model = train_model("Random Forest", X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Find detected intrusions
        test_df_with_predictions = test_df.copy()
        test_df_with_predictions['prediction'] = y_pred
        detected_intrusions = test_df_with_predictions[test_df_with_predictions['prediction'] == 'attack']

        report_content = f"""
 Automated Forensic Analysis Report
---

 1. Executive Summary
The report details the findings of an automated analysis of network traffic data from the NSL-KDD test dataset. The system identified a number of anomalies classified as potential intrusions.

 2. Dataset Analyzed
- Dataset: NSL-KDD Test Set
- Total Records Analyzed: {len(test_df):,}

 3. Intrusion Detection Model Performance
- Model Used: Random Forest Classifier (Default)
- Model Accuracy: {accuracy:.2%}
- Total Potential Intrusions Detected: {len(detected_intrusions):,}
- Total Normal Connections Identified: {len(test_df) - len(detected_intrusions):,}

 4. Analysis of Detected Intrusions
The following provides a breakdown of the original labels for the connections that the model flagged as 'attack'. This helps verify the model's findings against the ground truth.

"""
        # Breakdown of what the model called an 'attack'
        attack_breakdown = detected_intrusions['label'].value_counts().to_string()
        report_content += attack_breakdown
        
        report_content += """

## 5. Conclusion
The automated framework successfully analyzed the dataset and identified potential security incidents with high accuracy using the Random Forest model. Further investigation by a human analyst into the flagged connections is recommended.

---
End of Report
"""

        st.text_area("Generated Report", report_content, height=500)

        st.download_button(
            label="⬇️ Download Report as TXT",
            data=report_content,
            file_name="forensic_report.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()
