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
HOSTNAME = '10.69.10.122'
USERNAME = 'iman'
PASSWORD = os.environ['DATABASEPASSWORD']
DATABASE = 'dev'


def execute_query(query):
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

    column_list = [c[0] for c in cur.description]

    df_query = pd.DataFrame(cur.fetchall(), columns=column_list)

    LOGGER.debug("Size of dataset: {}".format(df_query.shape))
    connection.close()

    return df_query
