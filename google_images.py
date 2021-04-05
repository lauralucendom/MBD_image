#Imports Packages
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import requests
import time 
import urllib.request
import os
import platform
import shutil
import re

class GOOGLE_IMAGES:
    """Clase para la descarga de imágenes de Google."""
    def __init__(self):
        """
        Constructor de la clase
        """
        print("Instancia de la clase GOOGLE_IMAGES creada")

    def get_new_images(self, search_keyword = [], save_dir = 'archive/nuevas_imagenes', keywords = [' face', ' side face', ' looking up', ' looking down', ' wearning glasses', ' happy face', ' close up'], n_img = 4, visible = False):
        """
        Función que descarga imágenes de las personas públicas que recibe como parámetro.

        Args:
            search_keyword (list, optional): Lista de personas públicas que se quiera tener imágenes. Defaults to [].
            save_dir (str, optional): Carpeta de descarga. Defaults to 'archive/nuevas_imagenes'.
            keywords (list, optional): Palabras clave para las imágenes. Defaults to [' face', ' side face', ' looking up', ' looking down', ' wearning glasses', ' happy face', ' close up'].
            visible (bool, optional): Si se quiere ver como actúa el chromedriver se puede poner a True. Defaults to False.
        """
        if len(search_keyword) == 0:
            print("Se necesitan personas en el parámetro 'search_keyword'.")
        else:
            for s_keyword in search_keyword:
                if s_keyword != '':
                    # Directorio de trabajo
                    cwd = os.getcwd()
                    save_dir = os.path.join(cwd, save_dir)
                    # Chromedriver
                    chromeOptions = webdriver.chrome.options.Options()
                    chromeOptions.add_argument("--start-maximized")
                    if visible == False: chromeOptions.add_argument("--headless")
                    driver = webdriver.Chrome(executable_path="chromedriver.exe", chrome_options=chromeOptions)

                    driver.get('https://www.google.es/imghp?hl=en-GB&authuser=0&ogbl')
                    time.sleep(1)
                    driver.find_element_by_xpath('//*[@id="zV9nZe"]/div').click() # Aceptar las cookies

                    # Tratamos de crear las carpetas
                    try:
                        os.makedirs(save_dir + '/' + s_keyword)
                    except OSError as e:
                        if e.errno != 17:
                            raise   
                        pass
                    first = True
                    for keyword in keywords:
                        try:
                            pure_keyword = keyword
                            print("Descarga de imágenes de la búsqueda: {}{} HD".format(s_keyword, pure_keyword))
                            if first == True:
                                input_text = driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input')
                                first = False
                            else:
                                input_text = driver.find_element_by_xpath('//*[@id="REsRA"]')
                                input_text.clear()
                                time.sleep(2)
                            input_text.send_keys(s_keyword+pure_keyword+' HD')
                            input_text.send_keys(Keys.ENTER)
                            time.sleep(1)

                            for i in range(1,n_img+1):
                                try:
                                    lnk = driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').get_attribute('src')
                                    urllib.request.urlretrieve(lnk, "{}/{}/{}_{}.jpg".format(save_dir,s_keyword, keyword.replace(" ", ""), i))
                                except:
                                    pass
                        except:
                            driver.close()
                    driver.close()

    def move_new_images(self, last_number = 4, val_num = 1, dir_or = 'archive/nuevas_imagenes', dir_dest = 'archive/data/'):
        """
        Función que mueve las imágenes descargadas a las carpetas de entrenamiento del modelo

        Args:
            last_number (int, optional): Último número de las imágenes para pasarlo a validación. Defaults to 4.
            val_num (int, optional): Número de imágenes de cada keyword que queremos en validación. Defaults to 1.
            dir_or (str, optional): Direcctorio donde están las imágenes. Defaults to 'archive/nuevas_imagenes'.
            dir_dest (str, optional): Directorio dónde estan las imágenes para entrenar los modelos. Defaults to 'archive/data/'.
        """
        # Directorio de trabajo
        cwd = os.getcwd()
        dir_or = os.path.join(cwd, dir_or)
        dir_dest = os.path.join(cwd, dir_dest)
        print("Moviendo las imágenes descargadas")
        folders = os.listdir(dir_or)
        lista_val = [last_number - val for val in list(range(0, val_num))]
        for folder in folders:
            imagenes = os.listdir(dir_or+'/'+folder)
            for img in imagenes:
                if os.path.isdir(dir_dest+'/train/'+folder) == False: os.makedirs(dir_dest+'/train/'+folder)
                if os.path.isdir(dir_dest+'/val/'+folder) == False: os.makedirs(dir_dest+'/val/'+folder)
                if int(re.findall(r'\d+', img)[0]) in lista_val:
                    shutil.move(dir_or+'/'+folder+'/'+img, dir_dest+'/val/'+folder)
                else:
                    shutil.move(dir_or+'/'+folder+'/'+img, dir_dest+'/train/'+folder)
            os.rmdir(dir_or+'/'+folder)
        print("Movimiento de imágenes finalizado")