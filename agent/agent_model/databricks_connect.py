import torch
from databricks import sql
import numpy as np
import pandas as pd
from agent.config import DB_HOST, DB_PATH


def get_state(id, token, app_data):
    query = f'''
        select enterprise_id, state_tokenized, state
        from {app_data['table']}
        where {app_data['id']} = %(id)s
    '''

    print("Querying from database")
    conn = sql.connect(server_hostname=DB_HOST, http_path=DB_PATH, access_token=token)

    try:
        with conn.cursor() as cursor:
            cursor.execute(query, parameters={'id': id})
            print("Query executed on database")
            result = cursor.fetchall()[0].asDict()
            print("Result fetched from database")

        r = result['state_tokenized']
        sr = result['state']
        id = result['enterprise_id']
    except:
        r = []
        sr = []
        id = None

    if isinstance(r, str):
        state = np.array(eval(r)).astype('int')
        sr = np.array(eval(sr)).astype('int')
    elif isinstance(r, list):
        state = np.array(r).astype('int')
        sr = np.array(sr).astype('str')
    elif isinstance(r, np.ndarray):
        state = r.astype('int')
        sr = sr.astype('str')
    else:
        state = []
        sr = []

    state = torch.tensor(state)
    return state, sr, id
