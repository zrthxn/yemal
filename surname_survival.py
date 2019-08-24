# Surname survival simulator to predict
# the surnames of people after n number
# of generations

import numpy as np


class Person():
    surname = "NaN"
    married = False
    dead = False

    def create(surname):
        super.surname = surname

    def marry(spouse_surname):
        surname = spouse_surname

