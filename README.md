# Project description
I'm proud to share with you my Business Engineering bachelor's of engineering thesis project.
It contains automated NLP pipeline &amp; Web Scraper for Steam game reviews as well as analyzes player sentiment and topic modeling for Larian Studios to drive data-driven marketing and development decisions.
Based on thousands of user reviews collected from the platform, the thesis 
focuses on identifying and extracting the most frequently discussed topics among the 
studio’s customer base. Through sentiment analysis, the tool determines the general 
attitude of users toward Larian Studios’ products and highlights the issues that most strongly 
influence polarized opinions. The custom-built system, developed from scratch - including 
automation of the data acquisition process - not only enables data cleaning and preparation 
(preprocessing), but also allows for topic modeling, which significantly facilitates both 
product and marketing decision-making within the company. The conclusions drawn from 
the analysis have real business value, addressing the company’s managerial challenge 
of lacking an automated system for processing, predicting, and analyzing player opinions 
from Steam, one that would support accurate project and marketing decisions. The proposed 
solution demonstrates, among other things, that both the strongest praise and the harshest 
criticism from players stem from specific, well-defined aspects that, thanks to the custom 
tool, can finally be clearly presented to decision-makers. 

# What makes a difference
Comparing different models and algorithms to find the leader for this specific dataset. Solving a business problem starts from understanding everything about it. Understanding the nature of company's customer-base is a fundamental principle, hence said comparison - to address the problem swiftly and as accurately as possible.

# Tech stack and methodology
1) Fundamentals:
Language: Python
Environment: Jupyter Notebook
Base libraries: Pandas, NumPy
2) Web Scraping:
Source: Steam API
Tools: Requests, JSON, CSV
3) Natural Language Processing:
Preprocessing: NLTK (Tokenization, lemmatization, nose reduction (EDA))
Spam filtering: Scikit-learn (TF-IDF vectorization, SGDClassifier, Logistic Regression)
Sentiment analysis: VADER and LinearSVC from the lexical approaches, BERT (Transformers) and Flair from the deep learning ones
4) Forecasting and trend analysis:
Time-series modeling: Prophet, Exponential Smoothing
Regression&Machine Learning: XGBoost, LightGBM, Linear Regression
5) Visualization:
EDA: Matplotlib, Seaborn, Plotly
BI tools: Tableau
Deployment: PyInstaller 

# What does this repository contain
1) "requirements.txt" - file with Python packages requirements necessary to run the app
2) "inz_263807.pdf" - my bachelor's of engineering thesis which is a project documentation at the same time. It contains explanation of used technologies and evaluates received results
3) "models" file folder - it contains pre-trained machine learning and deep learning models used in the project
4) "data_generation.ipynb" - first part of the process' pipeline. Acquires necessary data
5) "data_processing.ipynb" - second part of the process' pipeline. Contains preprocessing data for further work and some of the EDA features. Produced plots in this part are not saved.
7) "spam_filter.ipynb" - third part of the process' pipeline. Comparing machine learning algorithms and picking the best one to remove unnecessary entries
8) "sentiment_analysis.ipynb" - fourth part of the process' pipeline. Comparing machine learning and deep learning algorithms to correctly assess sentiment
9) "absa_new_weights.ipynb" - fifth part of the process' pipeline. While the previous one was general, this one revolves around aspect based sentiment analysis
10) "review_prediction.ipynb" - sixth parth of the process' pipeline. Using gradient boosting algorithms and appropriate model reviewing metrics (such as MAE), this script predicts future (120 days) sentiment of client's reviews
12) "pos_neg_ratio.ipynb" - seventh part of the process' pipeline. It predicts future ratio of clients' reviews comparing regression models with others and then using the best one
13) "aplikacja.py" - every .ipynb file contained in a single .py file in form of an simple app. For those who like to skip theory for the results.
14) "Pulpity.twbx" - last part and a crown jewel of the process' pipeline. A Tableau file containing series of dashboards with plots full of crucial for business executives informations

# Visualizations and dashboards


# Instruction on how to run the "aplikacja.py" app
1) Initial requirements 
    - to run this app a Python version 3.12.10 is recommended. This is the one which was used
      in creating the project.
    - while downloading Python it is required to check "Add Python to PATH" option
    - to ensure that program will run correctly, it is advised to double-check whether the
      "models" folder is located in the same catalogue as the project's script.
    - for the consideration of optimalization the script downloads only 1000 records, which is
      an insufficient number to do proper analysis - if needed the amount can be influenced inside the code by changing the "MAX_REVIEWS_TO_FETCH" parameter's value. A           recommended one is 100 000
2) Environment configuration
    To prevent possible collisions and errors a virtual environment (venv) is recommended.
    - Step 1: Open the CMD inside the project's file folder
    - Step 2: Create virtual environment by typing:
                "python -m venv env"
    - Step 3: Acrivate the environment:
                ".\env\Scripts\activate"
    - Step 4: Install required packages by typing this in the CMD:
                "pip install -r requirements.txt"
3) Running the app
    When everything previously mentioned is prepare, please run the application by typing this in the CMD:
                 "python aplikacja.py"
4) Ending work with the project
    After ending work you can exit the window by pressing "X" or typing "exit" in the CMD. The products of this app's work should be three .hyper files intended to import into the "Pulpity.twbx" file to see pre-prepared visualizations 
