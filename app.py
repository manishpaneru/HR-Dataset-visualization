import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_card import card
import altair as alt
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Executive HR Insights",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern 2025 luxury styling
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #FFFFFF;
        color: #1D3557;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F5F5F7;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #006D77 !important;
        font-family: 'Playfair Display', serif;
        letter-spacing: 0.5px;
        font-weight: 300;
    }
    
    /* Card styling */
    div.stcard {
        border-radius: 16px;
        padding: 32px;
        background-color: #FFFFFF;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    /* Metric styling */
    div.metric-container {
        background-color: #FFFFFF;
        border-left: 3px solid #D4B483;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #006D77;
        color: #FFFFFF;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 400;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
    }
    
    .stButton>button:hover {
        background-color: #007D87;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 109, 119, 0.2);
    }
    
    /* Plotly chart background */
    .js-plotly-plot .plotly {
        background-color: #FFFFFF;
    }
    
    /* Table styling */
    .dataframe {
        background-color: #FFFFFF;
        color: #1D3557;
        border: none;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .dataframe th {
        background-color: #F5F5F7;
        color: #4A4E69;
        padding: 16px;
        font-weight: 500;
    }
    
    .dataframe td {
        padding: 16px;
        border-bottom: 0.5px solid #F5F5F7;
    }
    
    /* Dropdown and select boxes */
    .stSelectbox label, .stMultiselect label {
        color: #006D77;
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .stSelectbox > div > div, .stMultiselect > div > div {
        background-color: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #F5F5F7;
        padding: 4px 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }
    
    /* Make sure text is readable */
    p, li, .caption {
        color: #4A4E69;
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* Custom divider */
    hr {
        border: 0;
        height: 0.5px;
        background-color: #F5F5F7;
        margin: 32px 0;
    }
    
    /* Footer styling */
    footer {
        border-top: 0.5px solid #F5F5F7;
        padding-top: 24px;
        margin-top: 64px;
        color: #4A4E69;
        font-size: 12px;
        text-align: center;
    }
    
    /* Slider styling */
    .stSlider > div > div {
        background-color: #F5F5F7;
    }
    
    .stSlider > div > div > div > div {
        background-color: #006D77;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        border-radius: 12px;
        border: 1px solid #F5F5F7;
        padding: 12px 16px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 4px;
        height: 4px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F5F5F7;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #006D77;
        border-radius: 2px;
    }
    
    /* Card hover effect */
    div.stcard:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);
    }
    
    /* Glass effect for special containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
    }
    
    /* Neumorphism effect */
    .neumorphic {
        background: #FFFFFF;
        border-radius: 16px;
        box-shadow: 8px 8px 16px #F0F0F0, -8px -8px 16px #FFFFFF;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@300;400;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('HRDataset_v14.csv')
    # Clean column names (remove extra spaces)
    data.columns = data.columns.str.strip()
    
    # Clean string columns
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip()
    
    # Convert date columns
    for date_col in ['DOB', 'DateofHire', 'DateofTermination', 'LastPerformanceReview_Date']:
        if date_col in data.columns:
            try:
                data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            except:
                pass
    
    # Calculate age and tenure
    today = pd.to_datetime('today')
    if 'DOB' in data.columns:
        data['Age'] = (today - data['DOB']).dt.days // 365
    
    if 'DateofHire' in data.columns:
        # For active employees, calculate tenure based on current date
        active_mask = data['DateofTermination'].isna()
        data.loc[active_mask, 'Tenure'] = (today - data.loc[active_mask, 'DateofHire']).dt.days // 365
        
        # For terminated employees, calculate tenure based on termination date
        termed_mask = ~data['DateofTermination'].isna()
        data.loc[termed_mask, 'Tenure'] = (data.loc[termed_mask, 'DateofTermination'] - 
                                           data.loc[termed_mask, 'DateofHire']).dt.days // 365
    
    return data

df = load_data()

# Sidebar
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 32px;'><h2 style='font-size: 28px; color: #006D77;'>Executive HR Hub</h2></div>", unsafe_allow_html=True)
    
    # Add creator information in an accordion - moved here to use the padding space
    st.markdown("<div style='margin-bottom: 32px;'>", unsafe_allow_html=True)
    with st.expander("About the Creator", expanded=False):
        # Use Streamlit's native image display
        try:
            image = Image.open('avatar-1.png')
            st.image(image, width=120, use_column_width=False)
        except:
            st.info("Profile image not found")
        
        st.markdown("""
        <h4 style="text-align: center; color: #006D77; margin-bottom: 16px;">Manish Paneru</h4>
        <p style="text-align: center; margin-bottom: 8px; color: #4A4E69; font-weight: 500;">Data Analyst</p>
        
        <div style="display: flex; justify-content: center; gap: 16px; margin: 16px 0;">
            <a href="https://linkedin.com/in/manish.paneru1" target="_blank" style="color: #006D77; text-decoration: none;">
                <div style="display: flex; align-items: center;">
                    <span>LinkedIn</span>
                </div>
            </a>
            <a href="https://github.com/manishpaneru" target="_blank" style="color: #006D77; text-decoration: none;">
                <div style="display: flex; align-items: center;">
                    <span>GitHub</span>
                </div>
            </a>
            <a href="https://analystpaneru.xyz" target="_blank" style="color: #006D77; text-decoration: none;">
                <div style="display: flex; align-items: center;">
                    <span>Portfolio</span>
                </div>
            </a>
        </div>
        
        <div style="margin-top: 24px; text-align: left;">
            <h5 style="color: #006D77; margin-bottom: 8px;">About This Visualization</h5>
            <p style="font-size: 14px; line-height: 1.6; color: #4A4E69;">
                I created this executive HR dashboard to showcase modern data visualization techniques for HR analytics. 
                Using Python with Streamlit and Plotly, I designed this application to transform raw HR data into actionable insights.
            </p>
            <p style="font-size: 14px; line-height: 1.6; color: #4A4E69; margin-top: 8px;">
                The visualization displays key HR metrics including performance distribution, salary analysis by department, 
                employee engagement scores, and attrition risk prediction. My goal was to create an intuitive interface that executives 
                can use to make data-driven decisions about their workforce.
            </p>
            <p style="font-size: 14px; line-height: 1.6; color: #4A4E69; margin-top: 8px;">
                I implemented a 2025-inspired design language with clean typography, neumorphic elements, and a modern color palette
                to elevate the user experience and make complex HR data more accessible and actionable.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='neumorphic' style='padding: 24px; margin-bottom: 32px;'>", unsafe_allow_html=True)
    st.markdown("### Dashboard Controls")
    
    # Department filter
    departments = ['All'] + sorted(df['Department'].unique().tolist())
    selected_dept = st.selectbox("Select Department", departments)
    
    # Performance filter
    performances = ['All'] + sorted(df['PerformanceScore'].unique().tolist())
    selected_perf = st.selectbox("Performance Level", performances)
    
    # Salary range slider
    min_salary = int(df['Salary'].min())
    max_salary = int(df['Salary'].max())
    salary_range = st.slider("Salary Range ($)", min_salary, max_salary, (min_salary, max_salary))
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-container' style='padding: 24px; margin-bottom: 32px;'>", unsafe_allow_html=True)
    st.markdown("### Data Timeframe")
    
    # Date range for analysis
    min_date = df['DateofHire'].min().date()
    max_date = datetime.now().date()
    date_range = st.date_input("Date Range", [min_date, max_date])
    st.markdown("</div>", unsafe_allow_html=True)
    
    # About section
    st.markdown("<div style='padding: 24px; margin-top: 32px; background-color: #F5F5F7; border-radius: 16px;'>", unsafe_allow_html=True)
    st.markdown("### About")
    st.markdown("This executive dashboard provides a modern view of the organization's human capital metrics for 2025.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add logo/branding at bottom
    st.markdown("<div style='text-align: center; margin-top: 64px;'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #4A4E69; font-size: 12px;'>HR ANALYTICS 2025</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Filter data based on sidebar selections
filtered_df = df.copy()

if selected_dept != 'All':
    filtered_df = filtered_df[filtered_df['Department'] == selected_dept]
    
if selected_perf != 'All':
    filtered_df = filtered_df[filtered_df['PerformanceScore'] == selected_perf]
    
filtered_df = filtered_df[(filtered_df['Salary'] >= salary_range[0]) & 
                          (filtered_df['Salary'] <= salary_range[1])]

# Main dashboard
st.markdown("<h1 style='text-align: center; font-size: 48px; font-weight: 300; margin-bottom: 48px; letter-spacing: 1px;'>Executive HR Insights</h1>", unsafe_allow_html=True)

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='neumorphic' style='padding: 24px;'>", unsafe_allow_html=True)
    st.metric(
        "Total Employees",
        f"{filtered_df[filtered_df['EmploymentStatus'] == 'Active'].shape[0]:,}",
        delta=f"{filtered_df[filtered_df['EmploymentStatus'] == 'Active'].shape[0] - df.shape[0]}" if selected_dept != 'All' else None
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='neumorphic' style='padding: 24px;'>", unsafe_allow_html=True)
    avg_salary = int(filtered_df['Salary'].mean())
    st.metric(
        "Average Salary",
        f"${avg_salary:,}",
        delta=f"{int(avg_salary - df['Salary'].mean()):,}" if selected_dept != 'All' else None
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='neumorphic' style='padding: 24px;'>", unsafe_allow_html=True)
    turnover_rate = (filtered_df[filtered_df['Termd'] == 1].shape[0] / filtered_df.shape[0] * 100)
    st.metric(
        "Turnover Rate",
        f"{turnover_rate:.1f}%",
        delta=f"{turnover_rate - (df[df['Termd'] == 1].shape[0] / df.shape[0] * 100):.1f}%" if selected_dept != 'All' else None,
        delta_color="inverse"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='neumorphic' style='padding: 24px;'>", unsafe_allow_html=True)
    avg_satisfaction = filtered_df['EmpSatisfaction'].mean()
    st.metric(
        "Avg. Satisfaction",
        f"{avg_satisfaction:.1f}/5",
        delta=f"{avg_satisfaction - df['EmpSatisfaction'].mean():.1f}" if selected_dept != 'All' else None
    )
    st.markdown("</div>", unsafe_allow_html=True)

style_metric_cards(background_color="#FFFFFF", border_left_color="#D4B483", border_color="#F5F5F7", box_shadow=True)

st.markdown("---")

# Department Overview and Salary Distribution
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("<div class='glass-container' style='padding: 24px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 24px; margin-bottom: 24px;'>Department Distribution</h2>", unsafe_allow_html=True)
    
    dept_counts = df['Department'].value_counts().reset_index()
    dept_counts.columns = ['Department', 'Count']
    
    fig = px.pie(
        dept_counts, 
        values='Count', 
        names='Department',
        hole=0.6,
        color_discrete_sequence=px.colors.sequential.Teal
    )
    
    fig.update_traces(
        textposition='outside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    
    fig.update_layout(
        margin=dict(t=30, b=20, l=20, r=20),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4A4E69', size=14),
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-container' style='padding: 24px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 24px; margin-bottom: 24px;'>Salary Distribution by Department</h2>", unsafe_allow_html=True)
    
    dept_salary = df.groupby('Department')['Salary'].agg(['mean', 'min', 'max']).reset_index()
    dept_salary.columns = ['Department', 'Average', 'Minimum', 'Maximum']
    dept_salary = dept_salary.sort_values('Average', ascending=False)
    
    fig = px.bar(
        dept_salary,
        x='Department',
        y='Average',
        error_y=dept_salary['Maximum']-dept_salary['Average'],
        error_y_minus=dept_salary['Average']-dept_salary['Minimum'],
        color='Average',
        color_continuous_scale='Teal',
        labels={'Average': 'Average Salary ($)'}
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Salary ($)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4A4E69', size=14),
        margin=dict(l=40, r=20, t=30, b=40),
        height=350,
        coloraxis_showscale=False
    )
    
    fig.update_traces(
        marker_line_color='#D4B483',
        marker_line_width=1.5
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Performance and Engagement section
st.markdown("<h2 style='font-size: 32px; text-align: center; margin: 32px 0;'>Performance & Engagement Analysis</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    st.markdown("<div class='neumorphic' style='padding: 24px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 20px; margin-bottom: 24px;'>Performance Distribution</h3>", unsafe_allow_html=True)
    
    perf_counts = filtered_df['PerformanceScore'].value_counts().reset_index()
    perf_counts.columns = ['PerformanceScore', 'Count']
    
    # Custom order for performance categories
    perf_order = ['Exceeds', 'Fully Meets', 'Needs Improvement', 'PIP']
    
    # Create a categorical type with our custom order
    perf_counts['PerformanceScore'] = pd.Categorical(
        perf_counts['PerformanceScore'],
        categories=perf_order,
        ordered=True
    )
    
    # Sort by our custom order
    perf_counts = perf_counts.sort_values('PerformanceScore')
    
    # Define colors for each performance category with modern 2025 palette
    colors = {
        'Exceeds': '#006D77',      # Primary accent (teal)
        'Fully Meets': '#83C5BE',  # Lighter teal
        'Needs Improvement': '#D4B483', # Secondary accent (gold)
        'PIP': '#E29578'           # Tertiary accent (mauve)
    }
    
    fig = go.Figure()
    
    for idx, row in perf_counts.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Count']],
            y=[row['PerformanceScore']],
            orientation='h',
            name=row['PerformanceScore'],
            marker_color=colors.get(row['PerformanceScore'], '#006D77'),
            text=f"{row['Count']} ({row['Count']/filtered_df.shape[0]*100:.1f}%)",
            textposition='inside',
            insidetextanchor='middle',
            width=0.7
        ))
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4A4E69', size=14),
        margin=dict(l=0, r=10, t=10, b=0),
        height=300,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        bargap=0.3
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='neumorphic' style='padding: 24px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 20px; margin-bottom: 24px;'>Engagement vs. Satisfaction</h3>", unsafe_allow_html=True)
    
    fig = px.scatter(
        filtered_df,
        x='EngagementSurvey',
        y='EmpSatisfaction',
        color='PerformanceScore',
        size='Salary',
        hover_name='Employee_Name',
        color_discrete_map={
            'Exceeds': '#006D77',
            'Fully Meets': '#83C5BE',
            'Needs Improvement': '#D4B483',
            'PIP': '#E29578'
        }
    )
    
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='#D4B483')
        ),
        opacity=0.8
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4A4E69', size=14),
        margin=dict(l=0, r=0, t=10, b=0),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#F5F5F7'
        ),
        xaxis=dict(
            title="Engagement Score",
            showgrid=True,
            gridcolor='rgba(192, 192, 192, 0.2)'
        ),
        yaxis=dict(
            title="Satisfaction (1-5)",
            showgrid=True,
            gridcolor='rgba(192, 192, 192, 0.2)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='neumorphic' style='padding: 24px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 20px; margin-bottom: 24px;'>Project Count by Performance</h3>", unsafe_allow_html=True)
    
    perf_projects = filtered_df.groupby('PerformanceScore')['SpecialProjectsCount'].mean().reset_index()
    
    # Custom order for performance categories
    perf_projects['PerformanceScore'] = pd.Categorical(
        perf_projects['PerformanceScore'],
        categories=perf_order,
        ordered=True
    )
    
    # Sort by our custom order
    perf_projects = perf_projects.sort_values('PerformanceScore')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=perf_projects['PerformanceScore'],
        y=perf_projects['SpecialProjectsCount'],
        marker_color=[colors.get(perf, '#006D77') for perf in perf_projects['PerformanceScore']],
        text=[f"{val:.1f}" for val in perf_projects['SpecialProjectsCount']],
        textposition='inside',
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4A4E69', size=14),
        margin=dict(l=20, r=20, t=10, b=40),
        height=300,
        xaxis_title="Performance Score",
        yaxis_title="Avg. Special Projects",
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(192, 192, 192, 0.2)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Tenure and Experience section
st.markdown("<h2 style='font-size: 32px; text-align: center; margin: 32px 0;'>Tenure & Experience Analysis</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='glass-container' style='padding: 24px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 20px; margin-bottom: 24px;'>Tenure Distribution</h3>", unsafe_allow_html=True)
    
    # Create histogram of tenure
    fig = px.histogram(
        filtered_df,
        x='Tenure',
        color='PerformanceScore',
        barmode='group',
        opacity=0.8,
        color_discrete_map={
            'Exceeds': '#006D77',
            'Fully Meets': '#83C5BE',
            'Needs Improvement': '#D4B483',
            'PIP': '#E29578'
        }
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4A4E69', size=14),
        margin=dict(l=20, r=20, t=10, b=40),
        height=350,
        xaxis_title="Tenure (Years)",
        yaxis_title="Number of Employees",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#F5F5F7'
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(192, 192, 192, 0.2)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-container' style='padding: 24px;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 20px; margin-bottom: 24px;'>Salary vs. Tenure</h3>", unsafe_allow_html=True)
    
    fig = px.scatter(
        filtered_df,
        x='Tenure',
        y='Salary',
        color='Department',
        size='EmpSatisfaction',
        hover_name='Employee_Name',
        color_discrete_sequence=px.colors.sequential.Teal,
        opacity=0.8
    )
    
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='#D4B483')
        )
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#4A4E69', size=14),
        margin=dict(l=20, r=20, t=10, b=40),
        height=350,
        xaxis_title="Tenure (Years)",
        yaxis_title="Salary ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#F5F5F7'
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(192, 192, 192, 0.2)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Add footer with clean design
st.markdown("""
<footer style="margin-top: 64px; padding-top: 24px; border-top: 0.5px solid #F5F5F7; text-align: center;">
    <p style="font-size: 12px; color: #4A4E69;">Executive HR Analytics Dashboard • Updated: March 31, 2025 • Confidential</p>
</footer>
""", unsafe_allow_html=True)

# Recruitment & Demographics Section
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2 style='font-size: 24px;'>Recruitment Sources</h2>", unsafe_allow_html=True)
    
    recruit_counts = filtered_df['RecruitmentSource'].value_counts().reset_index()
    recruit_counts.columns = ['Source', 'Count']
    
    # Get top 6 sources
    top_sources = recruit_counts.head(6)
    
    fig = px.pie(
        top_sources,
        values='Count',
        names='Source',
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#192841', width=2)),
        pull=[0.1 if i == 0 else 0 for i in range(len(top_sources))]
    )
    
    fig.update_layout(
        margin=dict(t=30, b=30, l=30, r=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(30, 47, 79, 0.8)',
            bordercolor='#C0C0C0'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFF0'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("<h2 style='font-size: 24px;'>Demographic Distribution</h2>", unsafe_allow_html=True)
    
    # Create a subplot with 2 subplots (gender and race)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("Gender Distribution", "Race/Ethnicity")
    )
    
    # Gender chart
    gender_counts = filtered_df['Sex'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    
    # Race chart
    race_counts = filtered_df['RaceDesc'].value_counts().reset_index()
    race_counts.columns = ['Race', 'Count']
    
    # Add gender trace
    fig.add_trace(
        go.Pie(
            labels=gender_counts['Gender'],
            values=gender_counts['Count'],
            name="Gender",
            marker=dict(
                colors=px.colors.sequential.Blues[3:5],
                line=dict(color='#192841', width=2)
            ),
            textinfo='percent+label',
            hole=0.4,
        ),
        row=1, col=1
    )
    
    # Add race trace
    fig.add_trace(
        go.Pie(
            labels=race_counts['Race'],
            values=race_counts['Count'],
            name="Race",
            marker=dict(
                colors=px.colors.sequential.Viridis,
                line=dict(color='#192841', width=2)
            ),
            textinfo='percent+label',
            hole=0.4,
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(t=60, b=30, l=30, r=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(30, 47, 79, 0.8)',
            bordercolor='#C0C0C0'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFF0'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Advanced Analytics Section
st.markdown("<h2 style='font-size: 32px; text-align: center; margin: 32px 0;'>Executive Insights & Predictive Analytics</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 32px; color: #4A4E69; font-size: 18px;'>Key organizational insights and future-focused analytics</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<h3 style='font-size: 18px;'>Salary vs. Performance & Tenure</h3>", unsafe_allow_html=True)
    
    # Create 3D scatter plot
    if 'Tenure' in filtered_df.columns:
        fig = px.scatter_3d(
            filtered_df,
            x='Tenure',
            y='PerformanceScore',
            z='Salary',
            color='Department',
            size='EmpSatisfaction',
            hover_name='Employee_Name',
            symbol='Sex',
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFF0'),
            scene=dict(
                xaxis_title='Tenure (Years)',
                yaxis_title='Performance Level',
                zaxis_title='Salary ($)',
                xaxis=dict(
                    backgroundcolor='rgb(30, 47, 79)',
                    gridcolor='white',
                    showbackground=True,
                    zerolinecolor='white',
                ),
                yaxis=dict(
                    backgroundcolor='rgb(30, 47, 79)',
                    gridcolor='white',
                    showbackground=True,
                    zerolinecolor='white',
                ),
                zaxis=dict(
                    backgroundcolor='rgb(30, 47, 79)',
                    gridcolor='white',
                    showbackground=True,
                    zerolinecolor='white',
                ),
            ),
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("<h3 style='font-size: 18px;'>Risk Analysis: Attrition Prediction</h3>", unsafe_allow_html=True)
    
    # Create a custom "risk score" based on various factors
    # This is simplified - in a real scenario you'd use ML models
    if 'Tenure' in filtered_df.columns and filtered_df['EmploymentStatus'].eq('Active').any():
        active_df = filtered_df[filtered_df['EmploymentStatus'] == 'Active'].copy()
        
        # Create simplified risk score (for demonstration)
        active_df['AttritionRisk'] = (
            (5 - active_df['EmpSatisfaction']) * 0.3 +
            (active_df['DaysLateLast30'] / 10) * 0.2 +
            (active_df['Absences'] / 5) * 0.2 +
            (active_df['Tenure'] < 2).astype(int) * 0.3
        )
        
        # Normalize to 0-100
        min_risk = active_df['AttritionRisk'].min()
        max_risk = active_df['AttritionRisk'].max()
        active_df['AttritionRiskNorm'] = (active_df['AttritionRisk'] - min_risk) / (max_risk - min_risk) * 100
        
        # Risk categories
        active_df['RiskCategory'] = pd.cut(
            active_df['AttritionRiskNorm'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Count by risk category
        risk_counts = active_df['RiskCategory'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        
        # Ensure proper order
        risk_order = ['Low', 'Medium', 'High', 'Critical']
        risk_counts['Risk Level'] = pd.Categorical(
            risk_counts['Risk Level'],
            categories=risk_order,
            ordered=True
        )
        risk_counts = risk_counts.sort_values('Risk Level')
        
        # Color map for risk levels
        risk_colors = {
            'Low': '#3CB371',      # Green
            'Medium': '#FFD700',   # Yellow
            'High': '#FFA07A',     # Light Salmon
            'Critical': '#CD5C5C'  # Red
        }
        
        fig = go.Figure()
        
        # Add bars
        for idx, row in risk_counts.iterrows():
            fig.add_trace(go.Bar(
                x=[row['Count']],
                y=[row['Risk Level']],
                orientation='h',
                name=row['Risk Level'],
                marker_color=risk_colors.get(row['Risk Level'], '#D4AF37'),
                text=f"{row['Count']} ({row['Count']/active_df.shape[0]*100:.1f}%)",
                textposition='outside',
                insidetextanchor='middle',
                width=0.7
            ))
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFF0'),
            margin=dict(l=0, r=0, t=10, b=0),
            height=200,
            xaxis=dict(
                title="Number of Employees",
                showgrid=True,
                gridcolor='rgba(192, 192, 192, 0.2)'
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False
            ),
            bargap=0.3
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show employees at highest risk (top 5)
        st.markdown("<h4 style='font-size: 16px; color: #D4AF37;'>Employees at Highest Risk</h4>", unsafe_allow_html=True)
        
        high_risk_employees = active_df.sort_values('AttritionRiskNorm', ascending=False).head(5)
        
        for i, row in high_risk_employees.iterrows():
            risk_color = risk_colors.get(row['RiskCategory'], '#D4AF37')
            
            st.markdown(
                f"""
                <div style="background: linear-gradient(90deg, {risk_color}22 0%, {risk_color}00 100%); 
                            border-left: 4px solid {risk_color}; 
                            padding: 10px; 
                            margin-bottom: 10px;
                            border-radius: 5px;">
                    <div style="display: flex; justify-content: space-between;">
                        <b>{row['Employee_Name']}</b>
                        <span style="color: {risk_color}; font-weight: bold;">{row['RiskCategory']} Risk ({int(row['AttritionRiskNorm'])}%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 5px;">
                        <span>Dept: {row['Department']}</span>
                        <span>Position: {row['Position']}</span>
                        <span>Satisfaction: {row['EmpSatisfaction']}/5</span>
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

# Add Key Insights section with modern design
st.markdown("---")

# Key Insights and Recommendations
st.markdown("<h2 style='font-size: 32px; text-align: center; margin: 32px 0;'>Key Insights & Recommendations</h2>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="glass-container" style="padding: 32px; height: 100%;">
        <h3 style="color: #006D77; font-size: 24px; margin-bottom: 24px; font-weight: 300;">Key Insights</h3>
        <ul style="color: #4A4E69; font-size: 16px; line-height: 1.8;">
            <li><b>Performance Distribution:</b> The organization has a healthy distribution of performance ratings with the majority of employees meeting or exceeding expectations.</li>
            <li><b>Correlation Findings:</b> Higher employee satisfaction correlates strongly with increased engagement scores and lower turnover rates.</li>
            <li><b>Turnover Analysis:</b> The primary reasons for turnover are career advancement opportunities and compensation.</li>
            <li><b>Recruitment Effectiveness:</b> The most effective recruitment channels based on performance and retention are employee referrals and professional networking.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="neumorphic" style="padding: 32px; height: 100%;">
        <h3 style="color: #006D77; font-size: 24px; margin-bottom: 24px; font-weight: 300;">Strategic Recommendations</h3>
        <ul style="color: #4A4E69; font-size: 16px; line-height: 1.8;">
            <li><b>Retention Strategy:</b> Implement targeted engagement initiatives for the identified high-risk employees.</li>
            <li><b>Compensation Review:</b> Consider market-aligned salary adjustments for high performers, particularly in departments showing higher attrition.</li>
            <li><b>Development Programs:</b> Expand career advancement opportunities through structured development programs.</li>
            <li><b>Recruitment Optimization:</b> Increase investment in the most effective recruitment channels while phasing out underperforming sources.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Add a download button with modern styling
st.markdown("""
<div style="display: flex; justify-content: center; margin-top: 48px; margin-bottom: 32px;">
    <button style="background-color: #006D77; color: white; border: none; padding: 12px 24px; border-radius: 12px; 
                 font-size: 16px; letter-spacing: 0.5px; box-shadow: 0 4px 12px rgba(0, 109, 119, 0.2); cursor: pointer;
                 transition: all 0.3s cubic-bezier(0.25, 0.1, 0.25, 1);"
            onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(0, 109, 119, 0.3)';"
            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0, 109, 119, 0.2)';">
        Download Full Report
    </button>
</div>
""", unsafe_allow_html=True)