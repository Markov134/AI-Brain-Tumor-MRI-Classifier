from database import *
from preprocessing import *
from model import *
from train import *
from evaluate import *
from predict import *

def main():

    connection = connect_db()
    cursor = connection.cursor()
    cursor.execute("SELECT filepath FROM mri_images WHERE id = 3;")
    row = cursor.fetchone()

    cursor.close()
    connection.close()

    predict_image(row[0])



if __name__ == "__main__":
    main()