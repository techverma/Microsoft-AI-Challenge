import os

def implicit():
    from google.cloud import storage

    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    storage_client = storage.Client()

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)

def explicit_compute_engine(project):
    from google.auth import compute_engine
    from google.cloud import storage

    # Explicitly use Compute Engine credentials. These credentials are
    # available on Compute Engine, App Engine Flexible, and Container Engine.
    credentials = compute_engine.Credentials()

    # Create the client using the credentials and specifying a project ID.
    storage_client = storage.Client(credentials=credentials, project=project)

    # Make an authenticated API request
#     buckets = list(storage_client.list_buckets())
#     print(buckets)

private_key_file_name = "private_key.json"

working_dir = os.getcwd()

private_key_file_path = os.path.join(working_dir,private_key_file_name)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = private_key_file_path

#implicit()

project_key = '33e0b11b466d28ff6f7a2b863e6837739f96cb2f'
#explicit_compute_engine(project_key)

from google.cloud import translate
client = translate.Client()

import csv

new_file = open('../data/processed/train.txt', 'w')

loop = 0

translated_queries = []
with open('../data/processed/processed_train.csv', 'r') as csvfile:
    f = csv.reader(csvfile, delimiter=',')
    flag = True
    for row in f:
        loop += 1
        if loop%500 == 0:
            print(loop)
        if(flag):
            flag = False
            continue
        #print(row)
        result = client.translate(row[1],target_language='en', model='base')
        new_file.write(result['translatedText']+'__label__'+str(row[2])+'\n')
