import Qdrant
import datetime
# import sys
# import os
import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config.vectordb import delete_vector_collection
from .storage import vector_db_list
from config.vectordb import delete_vector_collection
from .storage import vector_db_list



def appendVectorName(collection_name: str):
    """ Append (collection_name, current_time) to vector_db_list """
    current_time = datetime.datetime.now()
    for entry in vector_db_list:
        if entry[0] == collection_name:
            vector_db_list.remove(entry)
    vector_db_list.append((collection_name, current_time))
    print(f"Appended: {collection_name}, {current_time}")  # Debugging

def trackVectorDBList():
    def giveTimeDiff(currtime: datetime.datetime, savetime: datetime.datetime):
        return (currtime - savetime).seconds // 60

    for entry in vector_db_list:
        collection_save_time = entry[1]
        if giveTimeDiff(datetime.datetime.now(), collection_save_time) >= 15:
            collection_name = entry[0]
            delete_vector_collection(qdrant_client=qdrant_client, collection_name=collection_name)
            vector_db_list.remove(entry)
            print(f"Removed: {collection_name}, it has overstayed its welcome (15 mins).")
            return f"Removed: {collection_name}, it has overstayed its welcome (15 mins)."
    
    print(f"Currently tracked vectorDBs: {vector_db_list}")
    return f"Currently tracked vectorDBs: {vector_db_list}"

def flushVectorDB():
    collections = qdrant_client.get_collections().collections
    if collections:
        print(f"Old Vector DBs found: {[col.name for col in collections]}")
        print("Flushing them all...")
        for collection in collections:
            delete_vector_collection(qdrant_client=qdrant_client, collection_name=collection.name)
    else:
        print("No old vector DBs found. Nothing to flush.")
