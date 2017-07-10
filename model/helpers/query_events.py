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

    LOGGER.debug("Size of dataset: %s", df_query.shape)
    connection.close()

    return df_query


def create_staging_and_insert_primary(table_name, records):
    """
    Function that creates a new temp table and populates it with records
    """

    connection = psycopg2.connect(host=HOSTNAME,
                                  user=USERNAME,
                                  password=PASSWORD,
                                  dbname=DATABASE,
                                  port=5439)
    cur = connection.cursor()
    args_str = ','.join(cur.mogrify("(%s,%s,%s,%s)", x).decode() for x in records)
    cur.execute("""

        CREATE TABLE {0}_staging (LIKE {0});

        INSERT INTO {0}_staging
                    (event_id, primary_classification, primary_score, uploaded_time)
               VALUES """.format(table_name) + args_str)

    connection.commit()

    cur.close()
    connection.close()


def create_staging_and_insert_secondary(table_name, records):
    """
    Function that creates a new temp table and populates it with records
    """

    connection = psycopg2.connect(host=HOSTNAME,
                                  user=USERNAME,
                                  password=PASSWORD,
                                  dbname=DATABASE,
                                  port=5439)
    cur = connection.cursor()
    args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s)", x).decode() for x in records)
    cur.execute("""

        CREATE TABLE {0}_staging (LIKE {0});

        INSERT INTO {0}_staging
                    (event_id, primary_classification, primary_score,
                    secondary_classification, secondary_score, uploaded_time)
               VALUES """.format(table_name) + args_str)

    connection.commit()

    cur.close()
    connection.close()


def create_staging_and_insert_tertiary(table_name, records):
    """
    Function that creates a new temp table and populates it with records
    """

    connection = psycopg2.connect(host=HOSTNAME,
                                  user=USERNAME,
                                  password=PASSWORD,
                                  dbname=DATABASE,
                                  port=5439)
    cur = connection.cursor()
    args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s)", x).decode() for x in records)
    cur.execute("""

        CREATE TABLE {0}_staging (LIKE {0});

        INSERT INTO {0}_staging
                    (event_id, primary_classification, primary_score,
                    secondary_classification, secondary_score,
                    tertiary_classification, tertiary_score, uploaded_time)
               VALUES """.format(table_name) + args_str)

    connection.commit()

    cur.close()
    connection.close()


def merge_records(table_name):
    """
    Merges records from the staging table to the original table
    """

    connection = psycopg2.connect(host=HOSTNAME,
                                  user=USERNAME,
                                  password=PASSWORD,
                                  dbname=DATABASE,
                                  port=5439)
    cur = connection.cursor()

    cur.execute("""

        DELETE FROM {0}
        USING {0}_staging
        WHERE {0}.event_id = {0}_staging.event_id;

        INSERT INTO {0}
        SELECT * FROM {0}_staging;

        DROP TABLE {0}_staging;

        """.format(table_name))

    connection.commit()
    cur.close()
    connection.close()


def upload_records(records):
    """
    Upserts records (list of lists) into primary classification table
    """

    connection = psycopg2.connect(host=HOSTNAME,
                                  user=USERNAME,
                                  password=PASSWORD,
                                  dbname=DATABASE,
                                  port=5439)

    cur = connection.cursor()
    args_str = ','.join(cur.mogrify("(%s,%s,%s,%s)", x).decode() for x in records)
    cur.execute("""
        INSERT INTO primary_classifications
                    (event_id, primary_classification, primary_score, uploaded_time)
               VALUES """ + args_str)

    connection.commit()

    cur.close()
    connection.close()
