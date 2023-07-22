import streamlit as st
import pandas as pd
from data import BuildModel
from data import CustomerAnalytics

# -----------------------------------------------------------------------------------------------
# initialize data model
customer_analy = CustomerAnalytics()

# -------------------------------------------------------------------------------------------------
# HEADER
st.title('Customer Analytics App')
st.image('./hr-image-small.png')
# ----------------------------------------------------------------------------------------------------------
# BODY
# COLUMNS
# Markdown ----------------------------------------------------------------------------------------------------------------------
st.markdown("""A common problem when creating models to generate business value from data is that 
            the datasets can be so large that it can take days for the model to generate predictions.""")
st.markdown('Hence, I have created a sample model to predict whether the students in the dataset'
            'are looking for a new job or not, information that they will then use to direct them to prospective recruiters')
# ------------------------------------------------------------------------------------------------------------------------------------
# CHARTS
# -----------------------------------------------------------------------------------------------------------------------------------------
st.subheader('**Visualization of the data**')
# Gender Representation plot --------------------------------------------------------
st.plotly_chart(customer_analy.gender_plot())
st.write('The Barplot above shows the split of the student population between'
         ' each gender category')
# -------------------------------------------------------------------------------------------
# Total working hours plot
st.plotly_chart(customer_analy.total_training_plot())
st.write('The Barplot above shows the breakdown of total working hours by each gender')
# ------------------------------------------------------------------------------------------------
# Education Level plot
st.plotly_chart(customer_analy.education_level_plot())
st.write('The Barplot above shows the Education level of the dataset')
# -----------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# Job Change plot
st.plotly_chart(customer_analy.job_change_plot())
st.write('The Barplot above indicates the job change by gender')
# Training Hours Boxplot
st.plotly_chart(customer_analy.training_hrs_plot())
st.write('The Boxplot above shows the distribution of training hours')
# -------------------------------------------------------------------------------------------------------------------
# SLIDER
st.sidebar.title('**User Input Features**')

# FORM attached to slider
with st.sidebar.form('User Input Features'):
    # Select Gender
    gender = st.selectbox(
        'Gender', ['Other', 'Male', 'Female'], key='gender')
    # Select Relevant Experience
    rel_exp = st.selectbox(
        'Relevant Experience', ['Has relevant experience', 'No relevant experience'], key='relevant_experience')
    # Select Enrolled Uni
    enrolled_uni = st.selectbox(
        'Enrolled University', ['Full time course',
                                'Part time course', 'no_enrollment'], key='enrolled_university')
    # Select Education Level
    educa_level = st.selectbox(
        'Education Level', ['Primary School', 'High School', 'Graduate', 'Masters', 'Phd'], key='education_level')
    # Select Major Discipline
    mj_disc = st.selectbox('Major Discipline', [
                           'Arts', 'Business Degree', 'Humanities', 'No Major', 'Other', 'STEM'], key='major_discipline')
    # Select Experience
    yr_exp = st.selectbox('Experience', [
                          '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '>20'], key='experience')
    # Select Company Size
    cmp_size = st.selectbox('Company Size', [
                            '<10', '10-49', '50-99', '100-499', '500-999', '1000-4999', '5000-9999', '10000+'], key='company_size')
    # Select Company Type
    comp_type = st.selectbox('Company Type', [
        'Early Stage Startup', 'Funded Startup', 'NGO', 'Other', 'Public Sector', 'Pvt Ltd'], key='company_type')
    # Select Length of last job
    lst_new_jb = st.selectbox('How long since last job?', [
                              'never', '1', '2', '3', '4', '>4'], help='Number of years between current and last job', key='last_new_job')
    # Select Number of training hours
    train_hrs = st.number_input(
        'Enter Number of Hours Worked', 1, 100000, key='training_hours')
    # ADD a submit button
    submiited = st.form_submit_button('Submit')
# -------------------------------------------------------------------------------------------------------------------------------------------------
cols = ['gender', 'relevant_experience', 'enrolled_university', 'education_level', 'major_discipline',
        'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours']

# Create dataframe from user inputs
df = pd.DataFrame(st.session_state.to_dict(), index=[0])
df = df[cols]

st.write('')
st.write('')


# Model Deployment
# -------------------------------------------------------------------------------------
model = BuildModel()
model_dt = model.load()
# Displays user input when submit button is clicked
if submiited:
    # Shows Data submited by user
    st.write('##### User Inputed data:')
    st.write(df)

    # --------------------------------------------
    # Makes prediction
    pred = model_dt.predict(df)
    if pred[0] == 0:
        st.write('###### Response based on model: ')
        st.success('*You are likely not to want to change job*')
        st.balloons()
    else:
        st.write('###### Response based on model: ')
        st.warning('*You are liekly to change job*')
