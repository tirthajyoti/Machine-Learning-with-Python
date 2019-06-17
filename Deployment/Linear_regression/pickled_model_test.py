import dill as pickle
import pandas as pd

filename = 'lm_model_v1.pk'
with open('models/'+filename ,'rb') as f:
    loaded_model = pickle.load(f)
print("Model loaded...")
print()

# Read test dataset
print("Reading test dataset\n... Done!\n")
test_df = pd.read_csv("data/housing_test.csv")
# Drop the first column
test_df.drop(test_df.columns[0],axis=1,inplace=True)

# Predictions using the loaded model
pred = loaded_model.predict(test_df.iloc[0:10])
print("Inference done...")
print()

# Print the results
print ("First 10 predictied prices are: ",[round(p,2) for p in pred[:10]])
print()
