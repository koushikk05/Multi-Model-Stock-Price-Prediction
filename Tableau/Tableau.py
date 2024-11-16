import pandas as pd

from tableauhyperapi import HyperProcess, Connection, TableDefinition, SqlType, Telemetry, Inserter, CreateMode, TableName



filename = pd.read_csv('/Users/koushikgovardhanam/Documents/Major/project/data/final_prediction.csv')

colnames =filename.columns

coltypes =filename.dtypes

field = { 
    'float64' :     SqlType.double(), 
    'float32' :     SqlType.double(),
    'int64' :       SqlType.double(),
    'int32' :       SqlType.double(),
    'object':       SqlType.text(),
    'bool':         SqlType.bool(),
    'datetime64[ns, UTC]':   SqlType.date(),
}


# for each column, add the appropriate info for the Table Definition
column_names = []
column_type = []
for i in range(0, len(colnames)):
    cname = colnames[i] #header of column
    coltype = coltypes[i] #pandas data type of column
    ctype = field.get(str(coltype)) #get corresponding sql column type 

    #store in lists to used for column and schema
    column_names.append(cname)
    column_type.append(ctype)


#name and path to save extract temporarily
PATH_TO_HYPER = 'hyper_extract.hyper'

# Step 1: Start a new private local Hyper instance
with HyperProcess(Telemetry.SEND_USAGE_DATA_TO_TABLEAU, 'myapp' ) as hyper:

# Step 2:  Create the the .hyper file, replace it if it already exists
    with Connection(endpoint=hyper.endpoint, 
                    create_mode=CreateMode.CREATE_AND_REPLACE,
                    database=PATH_TO_HYPER) as connection:

# Step 3: Create the schema(empty)
        connection.catalog.create_schema('Extract')


# Step 4: Create the table definition
        #defining the columns according to the dataframe
        cols = []
        for i, j in zip(column_names, column_type):
            columns = TableDefinition.Column(i, j)
            cols.append(columns)


        #creating schema 
        schema = TableDefinition(table_name=TableName('Extract','Extract'),
                columns= cols)

# Step 5: Create the table in the connection catalog
        connection.catalog.create_table(schema)
    
        with Inserter(connection, schema) as inserter:
            for index, row in filename.iterrows():
                inserter.add_row(row)
            inserter.execute()

    print("The connection to the Hyper file is closed.")     

