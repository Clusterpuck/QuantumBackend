import pyodbc

#NOTE Entire file no needed, delete later
class DatabaseConnector:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = pyodbc.connect(self.connection_string)
        self.cursors = []
    
    def create_cursor(self):
        cursor = self.connection.cursor()
        self.cursors.append(cursor)
        return cursor
    
    def close_cursor(self, cursor):
        position = self.cursors.index(cursor)
        self.cursors.pop(position)
        cursor.close()

    def close_all_cursors(self):
        if hasattr(self, 'cursors'):
            for cursor in self.cursors:
                cursor.close()

    #Function for connectionString
    def __connect_database(self):
        print()

    def __del__(self):
        if hasattr(self, 'close_all_cursors'):
            self.close_all_cursors()
        if hasattr(self, 'connection'):
            self.connection.close()