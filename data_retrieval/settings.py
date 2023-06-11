from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
load_dotenv()

DB_NAME = os.getenv('DB_NAME')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = '5432'
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_TYPE = 'postgres'

class Settings:

    def __init__(self,
        database_name=DB_NAME, database_user=DB_USER, 
        database_password=DB_PASSWORD, database_host=DB_HOST,
        database_port=DB_PORT, database_type=DB_TYPE):
        name = database_name
        user = database_user
        password = database_password
        host = database_host
        port = database_port
        type = database_type

        if type == 'postgres':
            type += 'ql'

        self.db_uri = '{type}://{user}:{password}@{host}:{port}/{name}'.format(
            type=type, user=user, password=password,
            host=host, port=port, name=name
        )

        # create database engine
        self.get_connection()

    def get_connection(self):
        self.db_engine = create_engine(self.db_uri)
