import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests  # For API calls
from utils import get_completion, system_prompt_HCM

# Set page configuration
st.set_page_config(layout="wide")

# this is the new data
df = pd.read_csv('Full_Synthetic_Employee_Data(All).csv', header=1, encoding='latin1')

# pre processing *(specific to the data)*
df['Durchschnittsgehalt Markt'] = df['Durchschnittsgehalt Markt'].str.replace(',', '').astype(int)
df['Durchschnittsgehalt nach Bildungsstand'] = df['Durchschnittsgehalt nach Bildungsstand'].str.replace(',', '').astype(int)
df['Durchschnittsgehalt nach Berufserfahrung'] = df['Durchschnittsgehalt nach Berufserfahrung'].str.replace(',', '').astype(int)
df['Durchschnittsgehalt nach Job Profil'] = df['Durchschnittsgehalt nach Job Profil'].str.replace(',', '').astype(int)
df['Durchschnittsgehalt nach Wohnort'] = df['Durchschnittsgehalt nach Wohnort'].str.replace(',', '').astype(int)
df['Gehalt'] = df['Gehalt'].str.replace(',', '').astype(int)

df['Arbeitsstunden Produktiv 1 (extern)'] = df['Arbeitsstunden Produktiv 1 (extern)'].str.replace(',', '').astype(int)
df['Arbeitsstunden Produktiv 2'] = df['Arbeitsstunden Produktiv 2'].str.replace(',', '').astype(int)
df['Arbeitsstunden Produktiv 3'] = df['Arbeitsstunden Produktiv 3'].str.replace(',', '').astype(int)
df['Arbeitsstunden Gesamt'] = df['Arbeitsstunden Gesamt'].str.replace(',', '').astype(int)


percentage_columns = [col for col in df.columns if df[col].astype(str).apply(lambda x: '%' in x).any()]
print(percentage_columns)
def convert_to_float(df, columns):
    for col in columns:
        df[col] = df[col].str.replace('%', '').astype(float) / 100
    return df

df = convert_to_float(df, percentage_columns)
# Define the weights for the weighted salary score
weights = {
    'Durchschnittsgehalt Markt': 0.10,  # 10%
    'Durchschnittsgehalt nach Bildungsstand': 0.15,  # 15%
    'Durchschnittsgehalt nach Berufserfahrung': 0.25,  # 25%
    'Durchschnittsgehalt nach Job Profil': 0.30,  # 30%
    'Durchschnittsgehalt nach Wohnort': 0.20,  # 20%
}

# Calculate the Weighted Salary Score
df['Weighted Salary Score'] = (
    df['Durchschnittsgehalt Markt'] * weights['Durchschnittsgehalt Markt'] +
    df['Durchschnittsgehalt nach Bildungsstand'] * weights['Durchschnittsgehalt nach Bildungsstand'] +
    df['Durchschnittsgehalt nach Berufserfahrung'] * weights['Durchschnittsgehalt nach Berufserfahrung'] +
    df['Durchschnittsgehalt nach Job Profil'] * weights['Durchschnittsgehalt nach Job Profil'] +
    df['Durchschnittsgehalt nach Wohnort'] * weights['Durchschnittsgehalt nach Wohnort']
)

# Function to plot salary vs weighted score
def plot_salary_vs_weighted_score(instance):
    salary = instance['Gehalt']
    weighted_score = instance['Weighted Salary Score']
    
    data = pd.DataFrame({
        'Category': ['Gehalt', 'Weighted Salary Score'],
        'Value': [salary, weighted_score]
    })
    
    plt.figure(figsize=(6, 5))
    sns.barplot(x='Category', y='Value', data=data, palette='viridis')
    plt.title('Gehalt vs Weighted Salary Score')
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.grid(True)
    st.pyplot(plt)

# Function to analyze salary trends
def analyze_salary(row):
    salary_data = {
        'Current Year': row['Gehalt'],
        'T-1 Year': row['Gehaltsentwicklung (Gehalt T - 1 Jahr)'],
        'T-2 Year': row['Gehaltsentwicklung (Gehalt T - 2 Jahr)'],
        'T-3 Year': row['Gehaltsentwicklung (Gehalt T - 3 Jahr)']
    }
    
    plt.figure(figsize=(6, 5))
    plt.plot(list(salary_data.keys()), list(salary_data.values()), marker='o')
    plt.title('Salary Trend Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Salary')
    plt.grid(True)
    st.pyplot(plt)
    
    comparison_data = {
        'Market Average': row['Durchschnittsgehalt Markt'],
        'Average by Education': row['Durchschnittsgehalt nach Bildungsstand'],
        'Average by Experience': row['Durchschnittsgehalt nach Berufserfahrung'],
        'Average by Job Profile': row['Durchschnittsgehalt nach Job Profil'],
        'Average by Location': row['Durchschnittsgehalt nach Wohnort']
    }
    
    plt.figure(figsize=(6, 5))
    plt.bar(comparison_data.keys(), comparison_data.values(), color='skyblue')
    plt.axhline(y=row['Gehalt'], color='r', linestyle='--', label='Current Salary')
    plt.title('Comparison with Averages')
    plt.xlabel('Category')
    plt.ylabel('Salary')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

# Function to perform clustering and plot results
def perform_clustering(df):
    # Calculate productivity ratios
    df['Produktivität 1 Ratio'] = df['Arbeitsstunden Produktiv 1 (extern)'] / df['Arbeitsstunden Gesamt']
    df['Produktivität 2 Ratio'] = df['Arbeitsstunden Produktiv 2'] / df['Arbeitsstunden Gesamt']
    df['Produktivität 3 Ratio'] = df['Arbeitsstunden Produktiv 3'] / df['Arbeitsstunden Gesamt']
    
    # Calculate average Zielerreichung
    df['Zielerreichung Mittelwert'] = df[['Zielerreichung (T - 1 Jahr)', 'Zielerreichung (T - 2 Jahr)', 'Zielerreichung (T - 3 Jahr)']].mean(axis=1)
    
    # Context-aware scaling
    df['Produktivität 1 Ratio Scaled'] = df['Produktivität 1 Ratio'] / 0.8  # Ideal = 0.8
    df['Produktivität 2 Ratio Scaled'] = df['Produktivität 2 Ratio'] / 0.9  # Ideal = 0.9
    df['Produktivität 3 Ratio Scaled'] = df['Produktivität 3 Ratio'] / 1.0  # Ideal = 1.0
    df['Zielerreichung Mittelwert Scaled'] = df['Zielerreichung Mittelwert'] / 1.0  # Ideal = 1.0
    df['Zielerreichung'] = df['Zielerreichung'] / 1.0  # Ideal = 1.0
    
    # Select scaled features for clustering
    features_scaled = ['Produktivität 1 Ratio Scaled', 'Produktivität 2 Ratio Scaled', 'Produktivität 3 Ratio Scaled', 'Zielerreichung Mittelwert Scaled', 'Zielerreichung']
    X_scaled = df[features_scaled]
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]  # First principal component
    df['PCA2'] = X_pca[:, 1]  # Second principal component
    
    return kmeans


# Automatically perform clustering when the app runs
kmeans = perform_clustering(df)

# Function to generate a textual summary of the data
def generate_summary(n=12):
    summary = "### Employee Data Analysis Report\n\n"
    
    # Salary Analysis Summary
    summary += "#### Salary Analysis\n"
    
    # Average Salary by Level
    level_salary = df.groupby('Level')['Gehalt'].mean().round(2)

    
  
    
    # Average Salary by Role
    role_salary = df.groupby('Rolle')['Gehalt'].mean().round(2)

    
    # Average Salary by Professional Experience
    experience_salary = df.groupby('Berufserfahrung')['Gehalt'].mean().round(2)

    
    # Clustering Analysis Summary
    summary += "#### Clustering Analysis\n"
    
    # Cluster Summary
    cluster_summary = df.groupby('Cluster').agg({
        'Gehalt': 'mean',
        'Level': 'mean',
        'Produktivität 1 Ratio': 'mean',
        'Produktivität 2 Ratio': 'mean',
        'Produktivität 3 Ratio': 'mean',
        'Zielerreichung Mittelwert': 'mean',
        'Zielerreichung': 'mean'
    }).round(2).reset_index()
    
    summary += "- **Cluster Summary:**\n"
    for _, row in cluster_summary.iterrows():
        summary += (
            f"  - **Cluster {row['Cluster']}:**\n"
            f"    - Average Salary: €{row['Gehalt']:.2f}\n"
            f"    - Average Level: {row['Level']:.2f}\n"
            f"    - Productivity 1 Ratio: {row['Produktivität 1 Ratio']:.2f}\n"
            f"    - Productivity 2 Ratio: {row['Produktivität 2 Ratio']:.2f}\n"
            f"    - Productivity 3 Ratio: {row['Produktivität 3 Ratio']:.2f}\n"
            f"    - Average Goal Achievement: {row['Zielerreichung Mittelwert']:.2f}\n"
            f"    - Goal Achievement: {row['Zielerreichung']:.2f}\n"
        )
    summary += "\n"
    
    # Individual Employee Summary (Example for the first employee)
    employee = df.iloc[employee_id]
    summary += "#### Individual Employee Summary\n"
    summary += (
        f"- **Employee Details:**\n"
        f"  - Level: {employee['Level']}\n"
        f"  - Education: {employee['Abschluss (höchster)']}\n"
        f"  - Role: {employee['Rolle']}\n"
        f"  - Professional Experience: {employee['Berufserfahrung']} years\n"
        f"  - Salary: €{employee['Gehalt']:.2f}\n"
        f"  - Cluster: {employee['Cluster']}\n"
    )
    
    # Compare Employee Salary with Averages
    level_avg = level_salary.get(employee['Level'], 0)
    education_avg = education_salary.get(employee['Abschluss (höchster)'], 0)
    role_avg = role_salary.get(employee['Rolle'], 0)
    experience_avg = experience_salary.get(employee['Berufserfahrung'], 0)
    
    summary += (
        f"- **Salary Comparison:**\n"
        f"  - Salary vs Level Average: €{employee['Gehalt']:.2f} (Employee) vs €{level_avg:.2f} (Level Average)\n"
        f"  - Salary vs Education Average: €{employee['Gehalt']:.2f} (Employee) vs €{education_avg:.2f} (Education Average)\n"
        f"  - Salary vs Role Average: €{employee['Gehalt']:.2f} (Employee) vs €{role_avg:.2f} (Role Average)\n"
        f"  - Salary vs Experience Average: €{employee['Gehalt']:.2f} (Employee) vs €{experience_avg:.2f} (Experience Average)\n"
    )
    
    return summary


# Streamlit app layout
st.title("Employee Data Analysis")

employee_id = st.number_input("Enter Employee ID", min_value=0, max_value=len(df)-1, value=12, step=1)

st.sidebar.header("Options")
option = st.sidebar.selectbox("Choose Analysis", ["Salary Analysis", "Clustering", "Report"])
# Function to predict cluster for new data
def predict_cluster(p1, p2, p3, z1, z2, kmeans):
    # Scale the input features
    p1_scaled = p1 / 0.8  # Ideal = 0.8
    p2_scaled = p2 / 0.9  # Ideal = 0.9
    p3_scaled = p3 / 1.0  # Ideal = 1.0
    z1_scaled = z1 / 1.0  # Ideal = 1.0
    z2_scaled = z2 / 1.0  # Ideal = 1.0
    
    # Predict the cluster
    new_data = [[p1_scaled, p2_scaled, p3_scaled, z1_scaled, z2_scaled]]
    cluster = kmeans.predict(new_data)
    return cluster[0]
if option == "Salary Analysis":
    st.header("Salary Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("Average Salary by Level:")
        level_salary = df.groupby('Level')['Gehalt'].mean()
        st.write(level_salary.round(2))
    
    with col2:
        plt.figure(figsize=(6, 3))
        sns.barplot(x=level_salary.index, y=level_salary.values, palette='viridis')
        plt.title('Average Salary by Level')
        plt.xlabel('Level')
        plt.ylabel('Average Salary')
        plt.grid(True)
        st.pyplot(plt)
    
    with col1:
        st.write("\nAverage Salary by Education:")
        education_salary = df.groupby('Abschluss (h\u00f6chster)')['Gehalt'].mean()
        st.write(education_salary.round(2))
    
    with col2:
        plt.figure(figsize=(6, 3))
        sns.barplot(x=education_salary.index, y=education_salary.values, palette='viridis')
        plt.title('Average Salary by Education')
        plt.xlabel('Education')
        plt.ylabel('Average Salary')
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(plt)
    
    with col1:
        st.write("\nAverage Salary by Role:")
        role_salary = df.groupby('Rolle')['Gehalt'].mean()
        st.write(role_salary.round(2))
    
    with col2:
        plt.figure(figsize=(6, 3))
        sns.barplot(x=role_salary.index, y=role_salary.values, palette='viridis')
        plt.title('Average Salary by Role')
        plt.xlabel('Role')
        plt.ylabel('Average Salary')
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(plt)
    
    with col1:
        st.write("\nAverage Salary by Professional Experience:")
        professional_experience_salary = df.groupby('Berufserfahrung')['Gehalt'].mean()
        st.write(professional_experience_salary.round(2))
    
    with col2:
        plt.figure(figsize=(6, 3))
        sns.lineplot(x=professional_experience_salary.index, y=professional_experience_salary.values, marker='o')
        plt.title('Average Salary by Professional Experience')
        plt.xlabel('Professional Experience (Years)')
        plt.ylabel('Average Salary')
        plt.grid(True)
        st.pyplot(plt)
    
    with col1:
        st.write("\nAverage Salary by Level and Education:")
        level_education_salary = df.groupby(['Level', 'Abschluss (h\u00f6chster)'])['Gehalt'].mean().unstack()
        st.write(level_education_salary.round(2))
    
    with col2:
        plt.figure(figsize=(10, 6))
        sns.heatmap(level_education_salary, annot=True, fmt=".0f", cmap='viridis')
        plt.title('Average Salary by Level and Education')
        plt.xlabel('Education')
        plt.ylabel('Level')
        st.pyplot(plt)
    
    with col1:
        st.write("\nAverage Salary by Level and Role:")
        level_role_salary = df.groupby(['Level', 'Rolle'])['Gehalt'].mean().unstack()
        st.write(level_role_salary.round(2))
    
    with col2:
        plt.figure(figsize=(10, 6))
        sns.heatmap(level_role_salary, annot=True, fmt=".0f", cmap='viridis')
        plt.title('Average Salary by Level and Role')
        plt.xlabel('Role')
        plt.ylabel('Level')
        st.pyplot(plt)
    
    with col1:
        st.write("\nAverage Salary by Education and Role:")
        education_role_salary = df.groupby(['Abschluss (h\u00f6chster)', 'Rolle'])['Gehalt'].mean().unstack()
        st.write(education_role_salary.round(2))
    
    with col2:
        plt.figure(figsize=(10, 6))
        sns.heatmap(education_role_salary, annot=True, fmt=".0f", cmap='viridis')
        plt.title('Average Salary by Education and Role')
        plt.xlabel('Role')
        plt.ylabel('Education')
        st.pyplot(plt)
    
    with col1:
        st.write("\nSalary vs Weighted Salary Score for the first employee:")
    
    with col2:
        plot_salary_vs_weighted_score(df.iloc[employee_id])
    
    with col1:
        st.write("\nSalary Trend Analysis for the first employee:")
    
    with col2:
        analyze_salary(df.iloc[employee_id])


elif option == "Clustering":
    st.header("Employee Clustering")
    
    # Plot the clusters
    plt.figure(figsize=(5, 4))
    plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis', marker='o')
    plt.title('2D PCA of Clusters (Context-Aware Scaling)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    st.pyplot(plt)
    
    # Analyze clusters
    cluster_summary = df.groupby('Cluster').agg({
        'Gehalt': 'mean',
        'Level': 'mean',
        'Produktivität 1 Ratio': 'mean',
        'Produktivität 2 Ratio': 'mean',
        'Produktivität 3 Ratio': 'mean',
        'Zielerreichung Mittelwert': 'mean',
        'Zielerreichung': 'mean'
    }).round(2).reset_index()
    
    st.write("Cluster Summary:")
    st.write(cluster_summary)
    
    # Predict cluster for new data
    st.sidebar.header("Predict Cluster")
    p1 = st.sidebar.number_input("Produktivität 1 Ratio", value=1.75)
    p2 = st.sidebar.number_input("Produktivität 2 Ratio", value=1.85)
    p3 = st.sidebar.number_input("Produktivität 3 Ratio", value=0.95)
    z1 = st.sidebar.number_input("Zielerreichung Mittelwert", value=0.9)
    z2 = st.sidebar.number_input("Zielerreichung", value=1.9)
    
    if st.sidebar.button("Predict Cluster"):
        
        cluster = predict_cluster(p1, p2, p3, z1, z2, kmeans)
        st.sidebar.write(f"Predicted Cluster: {cluster}")

elif option == "Report":
    st.header("Report")
    
    # Generate and display the summary
    summary = generate_summary(employee_id)
    st.markdown(summary)
    
    # GPT Recommendations
    st.write("### GPT Recommendations")
    if st.button("GPT Recommend"):
        recommendations =get_completion(system_prompt_HCM(generate_summary(employee_id)))
        st.write(recommendations)