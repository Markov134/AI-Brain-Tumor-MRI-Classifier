from database import *
from preprocessing import *
from model import *
from train import *
from evaluate import *
from predict import *

def main():

    """Image 1: """
    connection = connect_db()
    cursor = connection.cursor()
    cursor.execute("SELECT filepath FROM mri_images WHERE id = 3;")
    row = cursor.fetchone()

    connection.close()

    predict_image(row[0])

    """Image 2:"""
    connection = connect_db()
    cursor = connection.cursor()
    cursor.execute("SELECT filepath FROM mri_images WHERE id = 682;")
    row = cursor.fetchone()

    connection.close()

    predict_image(row[0])
    print(row[0])

    """Image 3:"""
    connection = connect_db()
    cursor = connection.cursor()
    cursor.execute("SELECT filepath FROM mri_images WHERE id = 1520;")
    row = cursor.fetchone()

    connection.close()

    predict_image(row[0])
    print(row[0])


if __name__ == "__main__":
    main()