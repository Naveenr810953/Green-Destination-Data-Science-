import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(file_path):
    """Load and verify the dataset"""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        print(f"Dataset shape: {df.shape}\n")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {str(e)}")
        return None

def analyze_attrition(df):
    """Main analysis function"""
    if df is None:
        return
    
    # 1. Calculate and display Attrition Rate
    print("\n" + "="*50)
    print("ATTIRITION ANALYSIS".center(50))
    print("="*50)
    
    attrition_rate = df['Attrition'].value_counts(normalize=True) * 100
    print(f"\nAttrition Rate:\n{attrition_rate.to_string()}\n")
    
    # Visualize attrition rate
    plt.figure(figsize=(8, 5))
    attrition_rate.plot(kind='bar', color=['#4CAF50', '#F44336'])
    plt.title('Employee Attrition Rate', pad=20)
    plt.xlabel('Attrition Status')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.show()
    
    # 2. Convert Attrition to numerical for analysis
    df['Attrition_num'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # 3. Analyze key factors
    print("\n" + "="*50)
    print("KEY FACTOR ANALYSIS".center(50))
    print("="*50)
    
    factors = ['Age', 'YearsAtCompany', 'MonthlyIncome']
    
    for factor in factors:
        analyze_factor(df, factor)
    
    # 4. Correlation analysis
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS".center(50))
    print("="*50)
    
    plot_correlation_matrix(df)
    
    # 5. Additional analyses
    print("\n" + "="*50)
    print("ADDITIONAL ANALYSES".center(50))
    print("="*50)
    
    plot_department_attrition(df)
    plot_job_satisfaction_attrition(df)
    plot_years_since_promotion(df)

def analyze_factor(df, feature):
    """Analyze distribution and statistical significance of a feature"""
    print(f"\nAnalyzing {feature}...")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Attrition', y=feature, data=df, 
                palette={'No': '#4CAF50', 'Yes': '#F44336'})
    plt.title(f'{feature} Distribution by Attrition Status', pad=20)
    plt.xlabel('Attrition Status')
    plt.ylabel(feature)
    plt.show()
    
    # Statistical test
    group1 = df[df['Attrition'] == 'No'][feature]
    group2 = df[df['Attrition'] == 'Yes'][feature]
    
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    
    print(f"T-test results for {feature}:")
    print(f"  - t-statistic: {t_stat:.2f}")
    print(f"  - p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("  - The difference is statistically significant (p < 0.05)")
        mean_diff = group1.mean() - group2.mean()
        print(f"  - Mean difference: {mean_diff:.2f}")
    else:
        print("  - The difference is not statistically significant")
    print()

def plot_correlation_matrix(df):
    """Plot correlation matrix for key variables"""
    print("\nPlotting correlation matrix...")
    
    corr_matrix = df[['Age', 'YearsAtCompany', 'MonthlyIncome', 'Attrition_num']].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix of Key Variables', pad=20)
    plt.show()
    
    print("\nCorrelation with Attrition:")
    print(corr_matrix['Attrition_num'].sort_values(ascending=False).to_string())

def plot_department_attrition(df):
    """Plot attrition by department"""
    print("\nAnalyzing attrition by department...")
    
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Department', hue='Attrition', data=df,
                 palette={'No': '#4CAF50', 'Yes': '#F44336'})
    plt.title('Attrition by Department', pad=20)
    plt.xlabel('Department')
    plt.ylabel('Number of Employees')
    plt.legend(title='Attrition Status')
    plt.show()
    
    # Calculate department-wise attrition rates
    dept_rates = df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack() * 100
    print("\nDepartment-wise Attrition Rates (%):")
    print(dept_rates.to_string())

def plot_job_satisfaction_attrition(df):
    """Plot attrition by job satisfaction level"""
    print("\nAnalyzing attrition by job satisfaction...")
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='JobSatisfaction', hue='Attrition', data=df,
                 palette={'No': '#4CAF50', 'Yes': '#F44336'})
    plt.title('Attrition by Job Satisfaction Level', pad=20)
    plt.xlabel('Job Satisfaction Level (1-4)')
    plt.ylabel('Number of Employees')
    plt.legend(title='Attrition Status')
    plt.show()
    
    # Calculate job satisfaction attrition rates
    js_rates = df.groupby('JobSatisfaction')['Attrition'].value_counts(normalize=True).unstack() * 100
    print("\nJob Satisfaction Attrition Rates (%):")
    print(js_rates.to_string())

def plot_years_since_promotion(df):
    """Plot attrition by years since last promotion"""
    print("\nAnalyzing attrition by years since last promotion...")
    
    plt.figure(figsize=(12, 6))
    sns.countplot(x='YearsSinceLastPromotion', hue='Attrition', data=df,
                 palette={'No': '#4CAF50', 'Yes': '#F44336'})
    plt.title('Attrition by Years Since Last Promotion', pad=20)
    plt.xlabel('Years Since Last Promotion')
    plt.ylabel('Number of Employees')
    plt.legend(title='Attrition Status')
    plt.show()

if __name__ == "__main__":
    # File path - adjust as needed
    file_path = 'greendestination.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found in the current directory.")
        print("Current directory contents:")
        print(os.listdir())
    else:
        # Load and analyze data
        df = load_data(file_path)
        if df is not None:
            analyze_attrition(df)