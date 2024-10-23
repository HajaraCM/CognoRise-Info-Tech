from flask import Flask, request, jsonify,render_template
import pandas as pd
import numpy as np
import joblib

model=joblib.load('model.pkl')
cluster=joblib.load('cluster.pkl')
scaler=joblib.load('scaler.pkl')
combined_df=pd.read_csv('combined_df.csv')
df=pd.read_csv('data_cluster.csv')


app = Flask(__name__)


   

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_fraud', methods=['POST','GET'])
def predict_fraud():

    result=None
    # Extract request data
    if request.method == 'POST':
        data = request.form

        amount = float(data['amount'])
        time_str = data['time']

        # Convert the time input (HH:MM) to a datetime object
        time_column = pd.to_datetime(time_str, format='%H:%M')

        # Extract hour and minute
        hour = time_column.hour
        minute = time_column.minute

   

        # Create a DataFrame for the new transaction
        new_transaction = pd.DataFrame({
            'Amount': [amount],
            'Hour': [hour],
            'Minute': [minute],
        })

        # Standardize the new transaction using the scaler
        new_transaction_scaled = scaler.transform(new_transaction)

        # Get the scaled hour, minute, and amount
        scaled_amount = new_transaction_scaled[0][0]
        scaled_hour = new_transaction_scaled[0][1]
        scaled_minute = new_transaction_scaled[0][2]

        # Check for existing transactions that match closely (old user)
        similar_transactions = df[
            (np.abs(df['Amount'] - scaled_amount) < 1e-5) &  # Use tolerance for Amount
            (np.abs(df['Hour'] - scaled_hour) < 1e-5) &  # Use scaled values and tolerance
            (np.abs(df['Minute'] - scaled_minute) < 1e-5)
        ]

        if not similar_transactions.empty:
            # Get PCA components from the most recent matching transaction
            pca_components = similar_transactions.loc[similar_transactions.index[-1], [f'V{i+1}' for i in range(28)]].values
        else:
            # Predict the cluster for the new transaction
            cluster_label = cluster.predict(new_transaction_scaled)[0]

            # Retrieve the members of the predicted cluster
            cluster_members = combined_df[combined_df['Cluster'] == cluster_label]

            if cluster_members.empty:
                return jsonify({"error": "No cluster members found for prediction."}), 404

            # Find the most similar transaction in the cluster based on hour, minute, and amount
            distances = np.abs(cluster_members['Hour'] - scaled_hour) + np.abs(cluster_members['Minute'] - scaled_minute) + np.abs(cluster_members['Amount'] - amount)
            similar_index = distances.idxmin()

            # Get PCA components from the most similar transaction
            pca_components = cluster_members.loc[similar_index, [f'V{i+1}' for i in range(28)]].values

        # Combine PCA components with the scaled hour, minute, and amount for the model input
        input_for_model = np.concatenate((pca_components, new_transaction_scaled[0]))

        # Predict using the Random Forest model
        prediction = model.predict([input_for_model])
        

        # Prepare the result as a JSON response
        result =  " Fraud detected: Please review the transaction for further verification." if prediction[0] == 1 else "No fraud detected: The transaction appears to be legitimate."

    return render_template('result.html',result=result)
        

 
if __name__ == '__main__':
    app.run(debug=True)
