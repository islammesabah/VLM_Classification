import sqlite3

def create_table(db_name):
    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(f'{db_name}')  
    cursor = conn.cursor()

    # Create the 'answers' table
    # Hyperparameters:
    # Include_memory: Whether to include the hisory of the conversation in the LLM input
    # Include_description: Whether to include the description of the class in the LLM input
    # Include_zero_shot_label: Whether to include the without-tree in the LLM input
    cursor.execute('''
        CREATE TABLE answers (
        Class TEXT,
        Sequence TEXT,
        Image_path TEXT,
        LLM_name TEXT,
        Include_memory TEXT DEFAULT "0",
        Include_description TEXT DEFAULT "0",
        Include_zero_shot_label TEXT DEFAULT "0",
        LLM_output_with_tree TEXT,
        LLM_output_without_tree TEXT,
        LLM_output_with_tree_class TEXT,
        LLM_output_without_tree_class TEXT,
        PRIMARY KEY (Image_path, LLM_name, Include_memory, Include_description, Include_zero_shot_label)
    ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

def main():

    db_name = input("Enter the name of the database that you need: ")
    if db_name[-3:] != ".db":
        db_name += ".db"
    create_table(db_name)

if __name__ == "__main__":
    main()
