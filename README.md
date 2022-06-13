# Module-13-Challenge - Venture Funding with Deep Learning
This application utilizes the knowledge of machine learning to create a binary classifier model using a deep neural network that will predict whether an applicant will become a successful business.

The Starter Code was provided along with the historical applicant data.

---

## User Story
Alphabet Soup’s (a venture capital firm) business team receives many funding applications from startups every day. This team has asked you to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

The business team has given you a CSV file containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. The CSV file contains a variety of information about each business, including whether or not it ultimately became successful. 

---

## Acceptance Criteria  
The application must meet the following acceptance criteria:  

* Pre-process the data for it to be used by the neural network model.
* Build a binary classification model using a neural network.
* Evaluate the performance of the model
* Optimize the neural network model by trying several strategies

---

## The Application  

The application follows these stages: 

### Pre-Process the Data for Use on a Neural Network Model

1. Read the `applicants_data.csv` data from the `Resources` folder into a Pandas DataFrame.
2. Review the dataframe and identify the categorical data that needs to be encoded. 
3. Drop the EIN and NAME columns.
4. Use `OneHotEncoder` method to encode the categorical columns and save the **encoded** data in a separate dataframe.
5. From the original DataFrame extract the **numerical columns** and add to the **encoded** dataframe.
6. Create the **Features (X)** and **Target (y)** datasets, with the Target being the **IS_SUCCESSFUL** column and the Features, the rest.
7. Split the data into training and testing datasets by using `train_test_split`.
8. Use `StandardScaler` to scale the Features dataset.

### Build a Binary Classification Model Using a Deep Neural Network
Use Tensorflow and Keras to build a deep neural network.
1. **Create** a Deep Neural Network by choosing number of hidden layers, assigning number of neurons to each hidden layer, number of features and output neurons 
2. **Compile** the model using `binary_crossentropy` loss function, the `adam` optimizer and the evauation metric of `accuracy`
3. **Fit** the model with the scaled training features data, target training data and `epochs` set to 50.

### Evaluate the Model

1. `Evaluate` the trained model with the test Features and Target datasets.
2. Save the model to a `HDF5` file.
3. Save the performance results in a `results` dataframe. 
4. Display the `results` dataframe

---

### Optimize the Deep Neural Network Model
Repeat the above steps of creating, compiling, fitting and evaluating by tuning one or more of the following:
* changing the number of layers
* changing the number of neurons in each layer
* dropping feature columns
* changing the loss function 
* changing activation functions 
* changing number of epochs

### Performance Report 
After 15 tries, tuning various parameters listed above reached the conclusion that it was difficult to improve the performance above accuracy of 73% with a minimum loss of 55% while using 'binary_crossentropy', and mean squared error (mse) and respective loss of 18% when the loss function of 'mean_squared_error' was utilized.

The details of the results are summarized in results_df dataframe, which can be viewed by [clicking here](performance_report.png)



## Technologies
The application is developed using:  
* Language: Python 3.7,   
* Packages/Libraries: Pandas; Dense from tensorflow.keras.layers; Sequential from tensorflow.keras.models; train_test_split from sklearn.model_selection; StandardScaler and OneHotEncoder from sklearn.preprocessing.
* Development Environment: VS Code and Terminal, Anaconda 2.1.1 with conda 4.11.0, Jupyterlab 3.2.9, Google Colab
* OS: Mac OS 12.1

---
## Installation Guide
Following are the instructions to install the application from its Github respository.  

### Clone the application code from Github as follows:
copy the URL link of the application from its Github repository      
open the Terminal window and clone as follows:  

   1. %cd to_your_preferred_directory_where_you want_to_store_this_application  
    
   2. %git clone URL_link_that_was_copied_in_step_1_above   
    
   3. %ls     
        Module-13-Challenge    
        
   4. %cd Module-13-Challenge     

At this point you will have the the entire application files in the current directory as follows:

    * README.md                       (this file that you are reading)  
    * performance_report.png          (model performance report
    * GC_venture_funding_with_deep_learning.ipynb         (the application jupter lab notebook)  
    * Resources                      (Folder with the data required) 
        - applicants_data.csv  
       
---

## Usage
The following details the instructions on how to run the application.  

### Setup Google Colab to run the application (on Mac computers)


   5. click on [Google Colab](https://colab.research.google.com/) - an IDE that allows you to run Jupyter Notebooks in the cloud. You can also copy and paste the link https://colab.research.google.com/ in your web browser as an alternative. 
    
   6. click on **upload** 
    
   7. click on **Choose File** to select the notebook **GC_venture_funding_with_deep_learning.ipynb** from the folder in Step 4 above 

---

#### Run the Application 

After the notebook **GC_venture_funding_with_deep_learning.ipynb** has loaded, click on **Runtime** tab and select **run all**  

After uploading all the required libraries, the notebook will be waiting for you to input the Resource file name.   
From the Resources folder in Step 4 above select the **applicants_csv** file and press enter 


### Setup the environment to run the application (on Windows)
This application was run only on a Mac Computer. However, if you wish to run it on a Windows machine, you will have to remove all the Colab references in the beginning where you upload the files in the notebook and then run the program as follows.  I have no way of testing on Windows, so didn't attempt to change the code.

Setup the environment using conda as follows:

    5. %conda create dev -python=3.7 anaconda  
    
    6. %conda activate dev  
    
    7. %jupyter lab  

---

#### Run the Application
THIS ASSUMES FAMILIARITY WITH JUPYTER LAB. If not, then [click here for information on Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/).  

After step 7 above, this will take you to the jupyter lab window, where you can open the application notebook **GC_venture_funding_with_deep_learning.ipynb** and run the application.  

**NOTE**:
>Your shell prompt will look something like __(dev) ashokpandey@Ashoks-MBP dir%__ ,  with:  
    - '(dev)' indicating the activated 'dev' environment,   
    - ' ashokpandey@Ashoks-MBP ' will be different for you as per your environment, and   
    - 'dir' directory is where the application is located.  
    - '%' sign is the shell prompt - it may be a dollar sign in your implementation 

---

## Contributors
Ashok Pandey - ashok.pragati@gmail.com   
www.linkedin.com/in/ashok-pandey-a7201237

---

## License
The source code is the property of the developer. The users can copy and use the code freely but the developer is not responsible for any liability arising out of the code and its derivatives.

---