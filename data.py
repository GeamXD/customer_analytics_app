import pandas as pd
import numpy as np
import plotly.express as px
from category_encoders import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle


class CustomerAnalytics:
    """_summary_

    Returns:
        _type_: _description_
        './dataset/customer_train.csv'
    """

    FILEPATH = '/home/geamxd/ds_repo/datacamp_ds/unguided/project_1/project__1/dataset/customer_train.csv'

    def __init__(self, wrangled=True):
        self.wrangled = wrangled

    def wrangle(self):
        """
        Takes a filepath and wrangles data according to my specification

        Parameter
        ---------
        filepath: str

        Returns
        --------
        returns: df a dataframe

        """
        filepath = self.FILEPATH
        mod = self.wrangled

        # Read csv file as pandas
        df = pd.read_csv(filepath)

        if mod:
            # Convert columns containing integers to int32
            df[['student_id', 'training_hours']] = (df.select_dtypes(np.int64)
                                                    .astype(np.int32))

            # Convert columns containg floats to float16
            df[['city_development_index', 'job_change']] = (df
                                                            .select_dtypes(np.float64)
                                                            .astype(np.float16))

            # Convert all necessary columns to category
            df[df.select_dtypes('object').columns] = (df
                                                      .select_dtypes('object')
                                                      .astype('category'))

            # Ordering categories in Company_size
            ord_1 = ['<10', '10-49', '50-99', '100-499', '500-999', '1000-4999',
                     '5000-9999', '10000+']
            df['company_size'] = (df['company_size'].cat
                                  .reorder_categories(new_categories=ord_1,
                                                      ordered=True))

            # Ordering categories in education_level
            ord_2 = ['Primary School', 'High School',
                     'Graduate', 'Masters', 'Phd']
            df['education_level'] = (df['education_level'].cat
                                     .reorder_categories(new_categories=ord_2,
                                                         ordered=True))

            # Ordering categories in last_new_job
            ord_3 = ['never', '1', '2', '3', '4', '>4']
            df['last_new_job'] = (df['last_new_job'].cat
                                  .reorder_categories(new_categories=ord_3,
                                                      ordered=True))

            # Ordering categories in experience
            ord_4 = ['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                     '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '>20']
            df['experience'] = (df['experience'].cat
                                .reorder_categories(new_categories=ord_4,
                                                    ordered=True))

            # Select only columns with experience 10yrs or more
            mask = df['experience'] >= '10'
            df = df[mask]

            # Select only colulmns with company size of at least 1000
            mask_2 = df['company_size'] .isin(
                ['10000+', '1000-4999', '5000-9999'])
            df = df[mask_2]

            # Handling missing values <= Threshold
            threshold = int(df.shape[0] * 0.05)
            cols_to_drop = df.columns[df.isna().sum() <= threshold]
            df.dropna(subset=cols_to_drop, inplace=True)

            # Renaming 'nan' Category to 'Other'
            df['gender'] = (df['gender'].astype(str)
                            .str.replace('nan', 'Other')
                            .astype('category'))

            # return dataframe
            return df
        else:
            return df

    def gender_plot(self):
        """
        Description:
            Barplot to show the different number of students by gender
        Returns:
            fig: a graph object
        """
        fig = px.bar(self.wrangle()['gender'].value_counts(
        ), title='Gender Representation')
        fig.update_layout(xaxis_title='Gender',
                          yaxis_title='Counts'
                          )
        fig.update_traces(showlegend=False)
        return fig

    def total_training_plot(self):
        """
        Description:
            Barplot to compare total working hours for each gender
        Returns:
            fig: a graph object
        """
        #
        fig = px.bar(data_frame=(self.wrangle().groupby('gender')['training_hours'].sum()).to_frame(),
                     y='training_hours', title='Gender breakdown of training hours')
        fig.update_layout(
            # title=dict(
            #     text='Total Training Hours vs Gender',
            #     font=dict(size=16),
            #     x=0.5,
            #     xref='paper'
            # ),
            xaxis_title='Gender',
            yaxis_title='Training Hours'
        )
        return fig

    def education_level_plot(self):
        """
        Description:
            Barplot to show the number of students per education level
        Returns:
            fig: a graph object
        """
        #
        fig = px.bar(self.wrangle()['education_level'].value_counts(
        ), title='Barplot of Education Level')
        fig.update_traces(showlegend=False)
        fig.update_layout(
            # title=dict(
            #     text='Barplot of Education Level',
            #     font=dict(size=16),
            #     x=0.5,
            #     xref='paper'
            # ),
            yaxis_title='Counts',
            xaxis_title='Education Level'
        )
        return fig

    def job_change_plot(self):
        """
        Description:
            barplot to show indicator of job change
        Returns:
            fig: a graph object
        """
        df = self.wrangle()
        df['job_change'] = df['job_change'].astype(bool)

        fig = px.bar(data_frame=df,
                     x='gender',
                     color='job_change',
                     barmode='group',
                     labels={'job_change': 'Job Change'},
                     title='Occupational mobility by gender')
        fig.update_layout(
            # title=dict(
            # text='Indication of Job change per Gender',
            # xref='paper',
            # x=0.5,
            # font=dict(size=16)),
            xaxis_title='Counts',
            yaxis_title='Gender'
        )
        return fig

    def training_hrs_plot(self):
        """_summary_
            Boxplot of Training Hours
        Returns:
            fig: a graph object
        """
        fig = px.box(data_frame=self.wrangle(), x='training_hours',
                     title='Boxplot of Training Hours')
        fig.update_layout(
            # title=dict(
            # text='Boxplot of Training Hours',
            # xref='paper',
            # x=0.5,
            # font=dict(size=16)),
            xaxis_title='Training Hours',
        )
        return fig


class BuildModel(CustomerAnalytics):
    """_summary_
            Builds model and other operations
    Args:
        CustomerAnalytics (_type_): _description_
    """

    def split(self):
        """_summary_
        """
        # Initialize a variable with dataframe
        df = super().wrangle()
        # Split data into training and testing datasets
        cols_to_drop = ['city', 'city_development_index',
                        'student_id', 'job_change']
        # Select target variable name
        target = 'job_change'
        # Drop columns
        X = df.drop(columns=cols_to_drop)
        # Instantiate target variable
        y = df[target]
        # Returns X_train, X_test, y_train, y_test
        return train_test_split(X, y, test_size=.2, random_state=42)

    def baseline(self):
        """_summary_
            Builds the models baseline features
        Returns:
            y_pred_baseline(int): accuracy score    
        """
        # Instantiates all the variables as instance attributes
        self.X_train, self.X_test, self.y_train, self.y_test = self.split()
        # Calculates naive model accuracy score
        y_preb_baseline = self.y_train.value_counts(normalize=True)[0] * 100
        # Returns naive model score
        return float(round(y_preb_baseline, 2))

    def __build(self):
        """_summary_
           Builds the models
        Returns:
            model: ml model    
        """
        # Instantiates all the variables as instance attributes
        self.X_train, self.X_test, self.y_train, self.y_test = self.split()
        # Creates a model using pipeline
        model = make_pipeline(OrdinalEncoder(), DecisionTreeClassifier(
            max_depth=3, random_state=42))
        # Fits the model and trains it
        model.fit(self.X_train, self.y_train)
        # Returns the trained model
        return model

    def evaluate(self):
        """_summary_
           Evaluates the model
        Returns:
            accuracy score(int): accuracy score    
        """
        # Instantiates all the variables as instance attributes
        self.X_train, self.X_test, self.y_train, self.y_test = self.split()
        # Instantiates the trained model
        model = self.__build()
        # Makes prediction
        y_pred = model.predict(self.X_test)
        # Returns accuracy score
        return float(round(accuracy_score(self.y_test, y_pred) * 100, 2))

    def make_prediction(self):
        """_summary_
        """
        # my_dict = {'gender': ['Other', 'Male', 'Female'],
        #            'relevant_experience': ['Has relevant experience', 'No relevant experience'],
        #            'enrolled_university': ['Full time course', 'Part time course', 'no_enrollment'],
        #            'education_level': ['Primary School', 'High School', 'Graduate', 'Masters', 'Phd'],
        #            'major_discipline': ['Arts', 'Business Degree', 'Humanities', 'No Major', 'Other', 'STEM'],
        #            'experience': ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '>20'],
        #            'company_size': ['<10', '10-49', '50-99', '100-499', '500-999', '1000-4999', '5000-9999', '10000+'],
        #            'company_type': ['Early Stage Startup', 'Funded Startup', 'NGO', 'Other', 'Public Sector', 'Pvt Ltd'],
        #            'last_new_job': ['never', '1', '2', '3', '4', '>4'],
        #            'training_hours': 'Any value'}

    def communitcate(self):
        """_summary_
        """
        pass

    def dump(self):
        """_summary_
            Saves the trained model
        Returns:
            _type_: _description_
        """
        model = self.__build()
        with open('cus_analy.pkl', 'wb') as f:
            dump = pickle.dump(model, f)
        return dump

    def load(self):
        """_summary_
                Loads model from saved location
        Returns:
            model: decision tree model
        """
        with open('./cus_analy.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
