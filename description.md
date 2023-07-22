# Customer Analytics: Preparing Data for Modeling
# Project Description
- Being able to create predictive models is very cool, but translating fancy models into real business value is a major challenge if the training data isn't stored efficiently.

- You've been hired by a major online data science training provider to store their data much more efficiently, so they can create a model that predicts if course enrollees are looking for a job. You'll convert data types, created ordered categories, and filter ordered categorical data so the data is ready for modeling.

## Instructions
The Head Data Scientist at Training Data Ltd. has asked you to create a DataFrame called ds_jobs_clean that stores the data in customer_train.csv much more efficiently. Specifically, they have set the following requirements:

- Columns containing integers must be stored as 32-bit integers (int32).
- Columns containing floats must be stored as 16-bit floats (float16).
- Columns containing nominal categorical data must be stored as the category data type.
- Columns containing ordinal categorical data must be stored as ordered categories, with an order that reflects the natural order of the column.
- The columns of ds_jobs_clean must be in the same order as the original dataset.
- The DataFrame should be filtered to only contain students with 10 or more years of experience at companies with at least 1000 employees, as their recruiter base is suited to more experienced professionals at enterprise companies.
- The final line must be ds_jobs_clean.head(100) to return only the first 100 rows.
