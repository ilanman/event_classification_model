"""

This module queries the database for events

"""

import os
import logging
import psycopg2
import pandas as pd


LOGNAME = 'log/event_classifier_log'
logging.basicConfig(filename=LOGNAME,
                    filemode='a',
                    format='%(asctime)s -  %(name)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


# connect to old cluster
HOSTNAME = '52.44.76.161'
USERNAME = 'iman'
PASSWORD = os.environ['DATABASEPASSWORD']
DATABASE = 'datawarehouseprod'


def execute_query(query, event_id):
    """
    Accepts query string and connection
    Returns a dataframe with results - exactly as it appears in the DB
    """

    connection = psycopg2.connect(host=HOSTNAME,
                                  user=USERNAME,
                                  password=PASSWORD,
                                  dbname=DATABASE,
                                  port=5439)
    # run query and store results in a dataframe
    LOGGER.info("Running query...")

    cur = connection.cursor()
    cur.execute(query)

    if event_id:

        df_query = pd.DataFrame(cur.fetchall(),
                                columns=['id', 'event_name', 'event_host',
                                         'event_subject', 'event_text',
                                         'created'])
    else:
        df_query = pd.DataFrame(cur.fetchall(),
                                columns=['id', 'p_class', 's_class', 't_class',
                                         'event_name', 'event_host',
                                         'event_subject', 'event_text',
                                         'created'])
    LOGGER.debug("Size of dataset: %s", df_query.shape)
    connection.close()

    return df_query
