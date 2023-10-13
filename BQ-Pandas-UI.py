from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
from pandasai import PandasAI
import pandas_gbq
from pandasai.llm.openai import OpenAI
from google.oauth2 import service_account
import os
import datetime
import json 

app = Flask(__name__, static_folder='static')
CORS(app)

credentials = service_account.Credentials.from_service_account_file('credentials.json')

project_id = 'smart-exchange-374615'  # Removed '.production'

tables = {
    "pages": f"`{project_id}.production.ga4_pages`",
    "devices": f"`{project_id}.production.ga4_devices`",
    "ecommerce_products": f"`{project_id}.production.ga4_ecommerce_products`",
    "channel": f"`{project_id}.production.ga4_channel`",
    "refferrals": f"`{project_id}.production.ga4_refferrals`",
}

dataframes = {}
customer_id = '1'  # Set customer ID here

for table_name, table in tables.items():
    sql = f"""
    SELECT
        *
    FROM
       {table}
    WHERE
       customer_id = '{customer_id}'
    """

    # Check if the cache file exists
    if not os.path.exists(f'cache/{table_name}_{customer_id}.pkl'):
        # If the cache file doesn't exist, load the data and save it to the cache file
        df = pandas_gbq.read_gbq(sql, project_id=project_id, credentials=credentials)

        # Fill NA values with -1 and convert to int64
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(-1).astype('int64')

        df.to_pickle(f'cache/{table_name}_{customer_id}.pkl')
    else:
        # If the cache file exists, load the data from the cache file
        df = pd.read_pickle(f'cache/{table_name}_{customer_id}.pkl')

    dataframes[table_name] = df

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/message', methods=['POST'])
def index():
    try:
        if request.method == 'POST':
            data = request.get_json()
            prompt = data.get('prompt')
            result = run_pandas_ai(dataframes, prompt)  
            return jsonify(result=result)
    except Exception as e:
        return jsonify(error=str(e)), 500


def run_pandas_ai(dataframes, prompt):
    llm = OpenAI(api_token="OPENAI_API")
    pandas_ai = PandasAI(llm, verbose=True, conversational=True)
    response = pandas_ai(list(dataframes.values()), prompt=prompt)
    return response

if __name__ == '__main__':
    app.run(debug=True)